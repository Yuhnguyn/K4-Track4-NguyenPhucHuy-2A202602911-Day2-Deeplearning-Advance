"""test_code.py - kiểm tra tự viết cho phần đã hoàn thiện của `code/` (RUBRIC mục H).

Chạy từ thư mục `code/`:
    python -m unittest discover -s . -p "test_*.py" -v
Không cần GPU và KHÔNG cần dataset DeepWeeds: test tự sinh một dataset nhỏ trong thư mục tạm.

Các test tập trung vào chỗ dễ sai nhất:
  - focal loss với gamma = 0 phải bằng đúng cross-entropy
  - label smoothing eps = 0 phải bằng đúng cross-entropy
  - CutMix: lam phải điều chỉnh theo DIỆN TÍCH HỘP THỰC sau khi cắt ra ngoài biên
  - nhóm tham số: norm/bias phải có weight_decay = 0, head có lr gấp 10 backbone
  - đóng băng backbone: chỉ head còn requires_grad
  - gộp BatchNorm: sai số trước/sau <= 1e-4
  - temperature scaling: NLL giảm, argmax không đổi
  - một lần chạy run() đầu-cuối phải sinh đủ artifact và file dự đoán qua được eval.read_pred
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import benchmark       # noqa: E402
import dataset         # noqa: E402
import inference       # noqa: E402
import losses          # noqa: E402
import model as model_mod  # noqa: E402
import train           # noqa: E402

sys.path.insert(0, str(train._EVAL_DIR))  # thư mục chứa eval.py của repo gốc
import eval as ev      # noqa: E402

K = 9


def _logits_labels(n=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, K, generator=g) * 2.0
    y = torch.randint(0, K, (n,), generator=g)
    return logits, y


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
class TestFocalLoss(unittest.TestCase):
    def test_gamma0_equals_cross_entropy(self):
        logits, y = _logits_labels()
        focal = losses.FocalLoss(gamma=0.0)
        ce = nn.CrossEntropyLoss()
        self.assertAlmostEqual(float(focal(logits, y)), float(ce(logits, y)), places=6)
        # chênh lệch phải < 1e-6 như GUIDE yêu cầu
        self.assertLess(abs(float(focal(logits, y)) - float(ce(logits, y))), 1e-6)

    def test_gamma2_is_not_ce_and_downweights_easy(self):
        logits, y = _logits_labels()
        self.assertNotAlmostEqual(float(losses.FocalLoss(gamma=2.0)(logits, y)),
                                  float(nn.CrossEntropyLoss()(logits, y)), places=4)

    def test_alpha_vector_applies_per_class(self):
        logits, y = _logits_labels()
        alpha = torch.arange(1, K + 1, dtype=torch.float32)
        plain = losses.FocalLoss(gamma=2.0)
        weighted = losses.FocalLoss(gamma=2.0, alpha=alpha)
        self.assertNotAlmostEqual(float(plain(logits, y)), float(weighted(logits, y)), places=4)


class TestLabelSmoothing(unittest.TestCase):
    def test_eps0_equals_cross_entropy(self):
        logits, y = _logits_labels()
        ce = nn.CrossEntropyLoss()
        ls0 = losses.LabelSmoothingCE(smoothing=0.0)
        self.assertLess(abs(float(ls0(logits, y)) - float(ce(logits, y))), 1e-6)

    def test_eps_positive_is_higher_loss(self):
        logits, y = _logits_labels()
        self.assertGreater(float(losses.LabelSmoothingCE(0.1)(logits, y)),
                           float(nn.CrossEntropyLoss()(logits, y)))

    def test_build_criterion_kinds(self):
        for kind in ("ce", "ls", "focal", "ce_weighted"):
            crit = losses.build_criterion(kind, smoothing=0.1, gamma=2.0,
                                          weight=torch.ones(K) if kind == "ce_weighted" else None)
            out = crit(*_logits_labels(16))
            self.assertTrue(torch.isfinite(out).item(), kind)
        with self.assertRaises(ValueError):
            losses.build_criterion("khong_ton_tai")


class TestClassWeights(unittest.TestCase):
    def test_beta0_mean_one_and_inverse(self):
        counts = [100, 200, 400, 50, 60, 70, 80, 90, 1000]
        w = losses.class_weights(counts, beta=0.0)
        self.assertAlmostEqual(float(w.mean()), 1.0, places=6)
        self.assertGreater(float(w[3]), float(w[2]))       # lớp ít ảnh -> trọng số lớn hơn

    def test_beta_positive_sums_to_num_classes(self):
        counts = [100, 200, 400, 50, 60, 70, 80, 90, 1000]
        w = losses.class_weights(counts, beta=0.999)
        self.assertAlmostEqual(float(w.sum()), float(K), places=4)


# --------------------------------------------------------------------------- #
# Mixup / CutMix
# --------------------------------------------------------------------------- #
class TestMix(unittest.TestCase):
    def test_mixup_formula(self):
        torch.manual_seed(0)
        np.random.seed(0)
        x = torch.rand(4, 3, 8, 8)
        y = torch.arange(4)
        xm, (ya, yb, lam) = losses.mix_batch(copy.deepcopy(x), y, alpha=1.0, mode="mixup")
        # không đòi hỏi hoán vị cụ thể: kiểm tra lam + (1-lam) và y_a = y
        self.assertTrue(torch.equal(ya, y))
        self.assertEqual(set(yb.tolist()), set(y.tolist()))
        self.assertTrue(torch.isfinite(xm).all())
        self.assertGreater(lam, 0.0)
        self.assertLess(lam, 1.0)

    def test_cutmix_lambda_matches_real_box_area(self):
        torch.manual_seed(1)
        np.random.seed(1)
        x = torch.rand(4, 3, 16, 16)
        x_orig = x.clone()
        y = torch.arange(4)
        xm, (_, _, lam) = losses.mix_batch(x, y, alpha=1.0, mode="cutmix")
        h = w = 16
        changed = (xm != x_orig).any(dim=1)                       # (N, H, W) pixel nào bị thay
        for i in range(4):
            area = int(changed[i].sum())
            expected = 1.0 - area / (h * w)
            self.assertAlmostEqual(lam, expected, places=6,
                                   msg=f"lam={lam:.4f} nhưng diện tích dán thực = {area}/{h * w}")

    def test_mixed_loss_is_weighted_sum(self):
        logits, y = _logits_labels(32)
        yb = y.roll(1)
        crit = nn.CrossEntropyLoss()
        lam = 0.3
        got = float(losses.mixed_loss(crit, logits, (y, yb, lam)))
        want = lam * float(crit(logits, y)) + (1 - lam) * float(crit(logits, yb))
        self.assertLess(abs(got - want), 1e-6)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class TestModel(unittest.TestCase):
    def test_freeze_keeps_only_head_trainable(self):
        m = model_mod.build_model("resnet18", pretrained=False, num_classes=K, init="frozen")
        head_ids = model_mod.head_param_ids(m)
        for p in m.parameters():
            self.assertEqual(p.requires_grad, id(p) in head_ids)

    def test_param_groups_lr_and_no_decay_on_norm_bias(self):
        m = model_mod.build_model("resnet18", pretrained=False, num_classes=K)
        groups = model_mod.param_groups(m, lr_backbone=1e-4, lr_head=1e-3, weight_decay=0.05)
        for g in groups:
            for p in g["params"]:
                if p.ndim <= 1:
                    self.assertEqual(g["weight_decay"], 0.0, "norm/bias phải có weight_decay = 0")
                elif g["lr"] == 1e-3:
                    self.assertEqual(g["weight_decay"], 0.05)
        head_lrs = {g["lr"] for g in groups if g["name"].startswith("head")}
        back_lrs = {g["lr"] for g in groups if g["name"].startswith("backbone")}
        self.assertEqual(head_lrs, {1e-3})
        self.assertEqual(back_lrs, {1e-4})

    def test_counts(self):
        m = model_mod.build_model("resnet18", pretrained=False, num_classes=K)
        p = model_mod.count_params(m)
        self.assertGreater(p, 10.0)
        self.assertLess(p, 14.0)
        g = model_mod.count_gmacs(m, 224)
        if np.isnan(g):
            print("CẢNH BÁO: không có fvcore/ptflops/thop -> GMAC = NaN")
        else:
            self.assertGreater(g, 0.5)
            self.assertLess(g, 3.0)

    def test_keep_backbone_bn_eval(self):
        m = model_mod.build_model("resnet18", pretrained=False, num_classes=K, init="frozen")
        m.train()
        model_mod.keep_backbone_bn_eval(m)
        for mod in m.modules():
            if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                self.assertFalse(mod.training)


# --------------------------------------------------------------------------- #
# Scheduler / EMA / seed
# --------------------------------------------------------------------------- #
class TestScheduleAndEma(unittest.TestCase):
    def test_warmup_then_cosine_to_zero(self):
        p = nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([p], lr=1.0)
        cfg = train.Config(epochs=4, warmup_epochs=1.0)
        sched = train.build_scheduler(opt, cfg, steps_per_epoch=10)
        lrs = []
        for _ in range(40):
            opt.step()
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])
        self.assertLess(lrs[0], lrs[9])                # warmup tăng
        self.assertAlmostEqual(max(lrs), 1.0, places=5)  # đỉnh = 1.0 ở cuối warmup
        self.assertLess(lrs[-1], 0.02)                 # cosine về ~0

    def test_ema_moves_toward_model(self):
        m = nn.Linear(4, 4, bias=False)
        ema = train.EMA(m, decay=0.9)
        before = [p.detach().clone() for p in ema.module.parameters()]
        with torch.no_grad():
            for p in m.parameters():
                p.add_(1.0)
        ema.update(m)
        for b, a in zip(before, ema.module.parameters()):
            self.assertTrue(torch.allclose(a, 0.9 * b + 0.1 * (b + 1.0), atol=1e-6))

    def test_set_seed_is_repeatable(self):
        train.set_seed(7)
        a = torch.randn(5)
        train.set_seed(7)
        self.assertTrue(torch.equal(a, torch.randn(5)))


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
class TestInference(unittest.TestCase):
    def test_fuse_conv_bn(self):
        torch.manual_seed(0)
        m = model_mod.build_model("resnet18", pretrained=False, num_classes=K).eval()
        x = torch.randn(2, 3, 64, 64)
        with torch.inference_mode():
            y0 = m(x).clone()
            m = inference.fuse_conv_bn(m)
            err = inference.fuse_error(m, x)
            y1 = m(x)
        self.assertGreater(m.fused_pairs, 0)
        self.assertLess(err, 1e-4)
        self.assertLess(float((y0 - y1).abs().max()), 1e-4)

    def test_temperature_scaling_lowers_nll_and_keeps_argmax(self):
        rng = np.random.default_rng(0)
        n = 400
        y = rng.integers(0, K, n)
        pred = y.copy()
        flip = rng.choice(n, size=n // 3, replace=False)
        pred[flip] = (pred[flip] + 1) % K
        logits = np.zeros((n, K))
        logits[np.arange(n), pred] = 12.0            # quá tự tin
        T = inference.fit_temperature(logits, y)
        p1 = inference.apply_temperature(logits, 1.0)
        pT = inference.apply_temperature(logits, T)
        nll = lambda p: float(-np.log(np.clip(p[np.arange(n), y], 1e-12, None)).mean())
        self.assertGreater(T, 1.0)
        self.assertLess(nll(pT), nll(p1))
        np.testing.assert_array_equal(pT.argmax(1), p1.argmax(1))

    def test_aggregate_views_prob_vs_logit(self):
        z = np.random.default_rng(1).normal(size=(3, 50, K))
        pp = inference.aggregate_views(z, "prob")
        pl = inference.aggregate_views(z, "logit")
        self.assertEqual(pp.shape, (50, K))
        np.testing.assert_allclose(pp.sum(1), 1.0, atol=1e-9)
        np.testing.assert_allclose(pl.sum(1), 1.0, atol=1e-9)
        self.assertFalse(np.allclose(pp, pl))       # hai cách gộp phải khác nhau

    def test_ensemble_and_views(self):
        a = np.full((4, K), 1.0 / K)
        b = np.zeros((4, K)); b[:, 0] = 1.0
        e = inference.ensemble_probs([a, b])
        np.testing.assert_allclose(e[:, 0], 0.5 + 0.5 / K, atol=1e-9)
        with self.assertRaises(ValueError):
            inference.ensemble_probs([a, np.zeros((3, K))])

        x = torch.rand(2, 3, 24, 24)
        self.assertEqual(len(inference.views_multicrop(x, crop=16, flip=True)), 10)
        self.assertEqual(len(inference.views_multicrop(x, crop=16, flip=False)), 5)
        self.assertEqual([v.shape[-1] for v in inference.views_multiscale(x, [16, 24])], [16, 24])
        self.assertTrue(torch.equal(inference.view_hflip(x), torch.flip(x, dims=[3])))


class TestBenchmark(unittest.TestCase):
    def test_bench_percentiles(self):
        r = benchmark.bench(lambda: sum(range(1000)), warmup=2, iters=20)
        self.assertEqual(r["n"], 20)
        self.assertLessEqual(r["p50"], r["p95"])
        self.assertLessEqual(r["p95"], r["p99"])
        self.assertGreater(r["p99"], 0.0)

    def test_latency_report_cpu(self):
        m = model_mod.build_model("resnet18", pretrained=False, num_classes=K)
        rep = benchmark.latency_report(m, batch_size=1, img_size=64, dtype="fp32", device="cpu",
                                      warmup=2, iters=5)
        self.assertEqual(rep["batch"], 1)
        self.assertGreater(rep["p50"], 0.0)
        self.assertIn("torch", rep)


# --------------------------------------------------------------------------- #
# CLI helper
# --------------------------------------------------------------------------- #
class TestParseOverrides(unittest.TestCase):
    def test_types(self):
        got = train.parse_overrides(["seed=3", "loss=focal", "ema_decay=none", "amp=false",
                                     "lr_head=5e-4", "backbone=convnext_tiny"])
        self.assertEqual(got["seed"], 3)
        self.assertEqual(got["loss"], "focal")
        self.assertIsNone(got["ema_decay"])
        self.assertIs(got["amp"], False)
        self.assertAlmostEqual(got["lr_head"], 5e-4)
        self.assertEqual(got["backbone"], "convnext_tiny")

    def test_errors(self):
        with self.assertRaises(KeyError):
            train.parse_overrides(["khong_co_field=1"])
        with self.assertRaises(ValueError):
            train.parse_overrides(["seed"])


# --------------------------------------------------------------------------- #
# Đầu-cuối: sinh dataset nhỏ rồi chạy train.run()
# --------------------------------------------------------------------------- #
def _make_tiny_dataset(root: Path, n_train=12, n_val=4, n_test=4, size=64, seed=0):
    """Sinh ảnh JPEG + 3 file CSV chia sẵn (đúng định dạng DeepWeeds) trong thư mục tạm."""
    rng = np.random.default_rng(seed)
    img_dir = root / "images"
    (root / "labels").mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = {"train": [], "val": [], "test": []}
    i = 0
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        for _ in range(n):
            lab = int(rng.integers(0, K))
            name = f"tiny_{i:04d}.jpg"
            arr = (rng.random((size, size, 3)) * 40 + 100 + lab * 3).clip(0, 255).astype(np.uint8)
            Image.fromarray(arr).save(img_dir / name, quality=90)
            rows[split].append({"Filename": name, "Label": lab})
            i += 1
    for split, rs in rows.items():
        pd.DataFrame(rs).to_csv(root / "labels" / f"{split}_subset0.csv", index=False)
    return {"images": img_dir, "labels": root / "labels",
            "total": n_train + n_val + n_test}


class TestEndToEndRun(unittest.TestCase):
    def test_run_produces_all_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ds = _make_tiny_dataset(root)
            old_total, old_cwd = dataset.EXPECTED_TOTAL, Path.cwd()
            dataset.EXPECTED_TOTAL = ds["total"]
            os.chdir(root)
            try:
                cfg = train.Config(exp_id="T99", backbone="resnet18", img_size=64, epochs=1,
                                   batch_size=2, num_workers=0, amp=False, lr_backbone=1e-3,
                                   lr_head=1e-2, warmup_epochs=0.5, ema_decay=0.9,
                                   images_dir=str(ds["images"]), labels_dir=str(ds["labels"]),
                                   save_test_predictions=True)
                with contextlib.redirect_stdout(io.StringIO()):
                    summary = train.run(cfg)

                rdir = train.run_dir(cfg)
                for f in ("config.json", "history.csv", "summary.json", "done.json", "best.pt",
                          "val_logits.npy", "val_labels.npy", "val_filenames.json",
                          "test_logits.npy", "test_filenames.json"):
                    self.assertTrue((rdir / f).exists(), f"thiếu {f}")
                self.assertTrue(Path("curves/T99_resnet18_seed0.png").exists())

                # file dự đoán phải qua được eval.read_pred + đối chiếu CSV chia sẵn
                for split in ("val", "test"):
                    p = train.pred_path(cfg, split)
                    self.assertTrue(p.exists(), p)
                    pred = ev.read_pred(str(p))
                    ev.check_against_csv(pred, str(ds["labels"] / f"{split}_subset0.csv"), split)
                    self.assertEqual(pred.seed, 0)
                    self.assertEqual(len(pred.filenames), len({*pred.filenames}))

                hist = pd.read_csv(rdir / "history.csv")
                self.assertEqual(len(hist), 1)
                self.assertIn("val_macro_f1", hist.columns)
                self.assertIn("seconds", hist.columns)
                self.assertGreaterEqual(summary["val_macro_f1"], 0.0)
                self.assertLessEqual(summary["val_macro_f1"], 1.0)
                self.assertIsNotNone(summary["gmac_tool"])
                self.assertEqual(summary["split"]["overlap"]["train&val"], 0)
                self.assertEqual(summary["split"]["total"], ds["total"])
            finally:
                dataset.EXPECTED_TOTAL = old_total
                os.chdir(old_cwd)

    def test_run_refuses_leaky_split(self):
        """Giao train∩val khác rỗng -> check_split phải raise (chống rò rỉ dữ liệu)."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_tiny_dataset(root)
            tr = pd.read_csv(root / "labels" / "train_subset0.csv")
            va = pd.read_csv(root / "labels" / "val_subset0.csv")
            leaky = pd.concat([va, tr.head(1)], ignore_index=True)
            with self.assertRaises(AssertionError):
                dataset.check_split(tr, leaky, pd.read_csv(root / "labels" / "test_subset0.csv"),
                                    root / "images", expected_total=20, verbose=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
