"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

ĐÃ HOÀN THIỆN từ starter/. MỘT hàm `run(cfg)` dùng chung cho mọi cấu hình (RUBRIC mục H):
đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0

Chỉ số dùng để chọn checkpoint (macro-F1 val) tính bằng `eval.compute_metrics` của repo gốc
để cùng định nghĩa với lúc chấm.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import dataset
import losses
import model as model_mod


# --------------------------------------------------------------------------- #
# Nhập eval.py của repo gốc (KHÔNG sửa eval.py)
# --------------------------------------------------------------------------- #
def _find_eval_dir() -> Path | None:
    """Tìm thư mục chứa eval.py: chính thư mục này, rồi ngược lên các cấp cha (bản sao y nguyên
    của repo nằm cạnh code/ hoặc ở gốc repo)."""
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "eval.py").exists():
            return parent
    return None


_EVAL_DIR = _find_eval_dir()
if _EVAL_DIR and str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

try:
    import eval as ev  # type: ignore  # module có tên trùng builtin, đây là eval.py của repo
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Không import được eval.py. Chép eval.py (nguyên bản, không sửa) cạnh code/ "
        "hoặc vào gốc repo, rồi chạy lại."
    ) from exc


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug | "a+b" ...
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    grad_clip: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False
    # hậu tố tên file dự đoán: None -> <exp_id>_seed<k>_<split>.csv ; "uncal" -> <exp_id>uncal_seed<k>_<split>.csv
    pred_tag: str | None = None


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    tag = getattr(cfg, "pred_tag", None)
    return Path(cfg.pred_dir) / f"{cfg.exp_id}{tag or ''}_seed{cfg.seed}_{split}.csv"


# --------------------------------------------------------------------------- #
# Tương thích phiên bản torch (Colab dùng torch >= 2.4 đổi API amp)
# --------------------------------------------------------------------------- #
def _autocast(enabled: bool):
    try:
        return torch.amp.autocast("cuda", enabled=bool(enabled))
    except (AttributeError, TypeError):  # pragma: no cover
        return torch.cuda.amp.autocast(enabled=bool(enabled))


def _make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=bool(enabled))
    except (AttributeError, TypeError):  # pragma: no cover
        return torch.cuda.amp.GradScaler(enabled=bool(enabled))


def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch < 1.13 không có weights_only
        return torch.load(path, map_location=device)


# --------------------------------------------------------------------------- #
# Seed / optimizer / scheduler / EMA
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên (random, numpy, torch CPU+CUDA) và seed worker DataLoader."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # benchmark=False + deterministic=True -> tích chập tất định, tái lập tốt hơn (chậm hơn chút).
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_optimizer(model, cfg: Config):
    """AdamW với các nhóm tham số của model.param_groups (backbone/head, norm+bias -> wd = 0)."""
    groups = model_mod.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    return torch.optim.AdamW(groups, lr=cfg.lr_head, betas=(0.9, 0.999), eps=1e-8)


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính `warmup_epochs` epoch rồi cosine về ~0 (slide trang 55).

    Cập nhật theo BƯỚC (iteration) — vẽ LR theo bước sẽ thấy đúng hình warmup + cosine.
    """
    warmup_steps = max(1, int(round(cfg.warmup_epochs * steps_per_epoch)))
    total_steps = max(warmup_steps + 1, int(cfg.epochs * steps_per_epoch))

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W  (slide trang 56).

    Buffer (running_mean/var của BatchNorm) được COPY từ model sống, không trung bình —
    đây là cách làm của timm (ModelEmaV2); trung bình running_var của các epoch khác nhau
    sẽ làm thống kê BN lệch khỏi giá trị đúng lúc suy luận.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        import copy
        if not 0.0 < decay < 1.0:
            raise ValueError(f"ema_decay phải trong (0, 1), nhận {decay}")
        self.decay = float(decay)
        self.module = copy.deepcopy(model)
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for pe, pm in zip(self.module.parameters(), model.parameters()):
            pe.mul_(d).add_(pm.detach(), alpha=1.0 - d)
        for be, bm in zip(self.module.buffers(), model.buffers()):
            if be.shape == bm.shape:
                be.copy_(bm)

    def copy_to(self, model: nn.Module) -> None:
        """Chép trọng số EMA vào `model` (nếu cần suy luận bằng chính đối tượng model)."""
        model.load_state_dict(self.module.state_dict())


# --------------------------------------------------------------------------- #
# Vòng huấn luyện / đánh giá
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện. Trả về {"train_loss", "lr", ...}."""
    model.train()
    if cfg.init == "frozen":
        model_mod.keep_backbone_bn_eval(model)      # BN của backbone đóng băng vẫn phải ở eval

    total, n = 0.0, 0
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if cfg.mix:
            x, targets = losses.mix_batch(x, y, alpha=cfg.mix_alpha, mode=cfg.mix)
        else:
            targets = None

        with _autocast(cfg.amp and device.type == "cuda"):
            logits = model(x)
            loss = losses.mixed_loss(criterion, logits, targets) if targets else criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(model)

        bs = x.size(0)
        total += float(loss.detach()) * bs
        n += bs

    return {"train_loss": total / max(n, 1), "lr": float(optimizer.param_groups[0]["lr"])}


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    """Chạy model trên một loader ở chế độ eval, KHÔNG tính gradient.

    Trả về (filenames: list[str], y_true: ndarray[N], logits: ndarray[N, 9], loss: float).
    Giữ đúng thứ tự của loader để ghép logit với tên file.
    """
    model.eval()
    names: list[str] = []
    ys: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    total, n = 0.0, 0
    for x, y, fname in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with _autocast(torch.cuda.is_available()):
            logits = model(x)
            loss = criterion(logits, y)
        names.extend(list(fname))
        ys.append(y.detach().cpu().numpy())
        zs.append(logits.detach().float().cpu().numpy())
        total += float(loss.detach()) * x.size(0)
        n += x.size(0)
    y_true = np.concatenate(ys) if ys else np.zeros(0, dtype=np.int64)
    logits = np.concatenate(zs) if zs else np.zeros((0, dataset.NUM_CLASSES), dtype=np.float32)
    return names, y_true, logits, total / max(n, 1)


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Vẽ đường cong training của một thí nghiệm -> curves/<exp_id>_<mota>.png (GUIDE.md mục 6.2).

    3 panel: (1) loss train/val, (2) macro-F1 val + top-1 val, (3) LR theo epoch.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(ep, [h["train_loss"] for h in history], marker="o", ms=3, label="train loss")
    ax[0].plot(ep, [h["val_loss"] for h in history], marker="s", ms=3, label="val loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].grid(alpha=.3)

    ax[1].plot(ep, [h["val_macro_f1"] for h in history], marker="o", ms=3,
               color="tab:green", label="macro-F1 val")
    ax[1].plot(ep, [h["val_top1"] for h in history], marker="s", ms=3,
               color="tab:orange", label="top-1 val")
    best = max(range(len(history)), key=lambda i: history[i]["val_macro_f1"])
    ax[1].axvline(history[best]["epoch"], ls="--", lw=1, color="grey",
                  label=f"best epoch {history[best]['epoch']}")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("chỉ số (val)"); ax[1].legend(); ax[1].grid(alpha=.3)

    ax[2].plot(ep, [h["lr"] for h in history], marker="o", ms=3, color="tab:red")
    ax[2].set_xlabel("epoch"); ax[2].set_ylabel("LR (nhóm đầu)"); ax[2].grid(alpha=.3)

    fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# run()
# --------------------------------------------------------------------------- #
def _version_info() -> dict:
    import timm
    info = {"torch": torch.__version__, "timm": timm.__version__, "python": sys.version.split()[0]}
    try:
        import torchvision
        info["torchvision"] = torchvision.__version__
    except Exception:  # pragma: no cover
        info["torchvision"] = None
    return info


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt."""
    t_start = time.perf_counter()
    set_seed(cfg.seed)
    rdir = run_dir(cfg)
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), "utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. dữ liệu + kiểm tra chia (S1-S6)
    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, cfg.fold)
    split_info = dataset.check_split(train_df, val_df, test_df, cfg.images_dir, verbose=False,
                                     expected_total=dataset.EXPECTED_TOTAL)

    # 3. model + transform + loader
    model = model_mod.build_model(cfg.backbone, pretrained=(cfg.init != "scratch"),
                                 num_classes=dataset.NUM_CLASSES, drop_rate=cfg.drop_rate,
                                 init=cfg.init)
    pcfg = getattr(model, "pretrained_cfg", None) or {}
    mean = tuple(pcfg.get("mean", dataset.IMAGENET_MEAN))
    std = tuple(pcfg.get("std", dataset.IMAGENET_STD))
    model.to(device)

    tf_train = dataset.build_transforms(True, cfg.img_size, cfg.aug, mean, std)
    tf_eval = dataset.build_transforms(False, cfg.img_size, cfg.aug, mean, std)
    train_loader = dataset.make_loader(train_df, cfg.images_dir, tf_train, cfg.batch_size,
                                      train=True, sampler=cfg.sampler, num_workers=cfg.num_workers,
                                      seed=cfg.seed)
    val_loader = dataset.make_loader(val_df, cfg.images_dir, tf_eval, cfg.batch_size,
                                    train=False, sampler=None, num_workers=cfg.num_workers,
                                    seed=cfg.seed)

    # 4. criterion / optimizer / scheduler / scaler / EMA
    counts = train_df["Label"].value_counts().reindex(range(dataset.NUM_CLASSES), fill_value=0).to_numpy()
    weight = None
    if cfg.loss == "ce_weighted" or cfg.class_weight_beta is not None:
        weight = losses.class_weights(counts, beta=cfg.class_weight_beta or 0.0)
    criterion = losses.build_criterion(cfg.loss, smoothing=cfg.label_smoothing,
                                       gamma=cfg.focal_gamma, weight=weight).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    amp_on = bool(cfg.amp) and device.type == "cuda"
    scaler = _make_scaler(amp_on)
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None
    eval_model = ema.module if ema is not None else model

    # 5. vòng epoch: chọn checkpoint theo MACRO-F1 VAL (hòa thì lấy epoch sớm hơn -> chỉ thay khi >)
    history: list[dict] = []
    best = {"macro_f1": -1.0, "epoch": -1, "top1": float("nan")}
    best_path = rdir / "best.pt"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, cfg.epochs + 1):
        te = time.perf_counter()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg,
                             device, ema=ema)
        vnames, vy, vz, vloss = evaluate(eval_model, val_loader, criterion, device)
        vprobs = torch.softmax(torch.as_tensor(vz, dtype=torch.float32), dim=1).numpy()
        vm = ev.compute_metrics(vy, vprobs.argmax(1), vprobs)
        sec = time.perf_counter() - te
        history.append({
            "epoch": epoch, "train_loss": tr["train_loss"], "val_loss": vloss,
            "val_macro_f1": vm["macro_f1"], "val_top1": vm["top1"],
            "val_balanced_acc": vm["balanced_acc"], "val_ece": vm["ece"], "lr": tr["lr"],
            "seconds": sec,
        })
        print(f"  [epoch {epoch:>2}/{cfg.epochs}] train_loss {tr['train_loss']:.4f} | "
              f"val_loss {vloss:.4f} | val macro-F1 {vm['macro_f1']:.4f} | val top-1 {vm['top1']:.4f} | "
              f"{sec:.1f}s", flush=True)

        if vm["macro_f1"] > best["macro_f1"]:
            best = {"macro_f1": float(vm["macro_f1"]), "epoch": epoch, "top1": float(vm["top1"]),
                    "val_loss": float(vloss), "val_balanced_acc": float(vm["balanced_acc"]),
                    "val_ece": float(vm["ece"]), "val_nll": float(vm["nll"])}
            torch.save({"state_dict": eval_model.state_dict(), "cfg": asdict(cfg), "epoch": epoch},
                       best_path)

    # 6. nạp checkpoint tốt nhất -> logit/dự đoán trên VAL
    ckpt = _torch_load(best_path, device)
    eval_model.load_state_dict(ckpt["state_dict"])
    vnames, vy, vz, _ = evaluate(eval_model, val_loader, criterion, device)
    vprobs = torch.softmax(torch.as_tensor(vz, dtype=torch.float32), dim=1).numpy()
    np.save(rdir / "val_logits.npy", vz)
    np.save(rdir / "val_labels.npy", vy)
    (rdir / "val_filenames.json").write_text(json.dumps(vnames), "utf-8")
    ev.save_predictions(pred_path(cfg, "val"), vnames, vy, vprobs)

    # 7. TEST: chỉ một lần, chỉ khi cfg.save_test_predictions (Bước 4, quy tắc S4)
    test_info = None
    if cfg.save_test_predictions:
        test_loader = dataset.make_loader(test_df, cfg.images_dir, tf_eval, cfg.batch_size,
                                         train=False, sampler=None, num_workers=cfg.num_workers,
                                         seed=cfg.seed)
        tnames, ty, tz, tloss = evaluate(eval_model, test_loader, criterion, device)
        tprobs = torch.softmax(torch.as_tensor(tz, dtype=torch.float32), dim=1).numpy()
        np.save(rdir / "test_logits.npy", tz)
        np.save(rdir / "test_labels.npy", ty)
        (rdir / "test_filenames.json").write_text(json.dumps(tnames), "utf-8")
        ev.save_predictions(pred_path(cfg, "test"), tnames, ty, tprobs)
        tm = ev.compute_metrics(ty, tprobs.argmax(1), tprobs)
        test_info = {"top1": tm["top1"], "macro_f1": tm["macro_f1"], "ece": tm["ece"],
                     "nll": tm["nll"], "loss": float(tloss)}

    # 8. history.csv, biểu đồ, summary
    import pandas as pd
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(rdir / "history.csv", index=False)
    plot_curves(history, Path("curves") / f"{cfg.exp_id}_{cfg.backbone}_seed{cfg.seed}.png",
                f"{cfg.exp_id} · {cfg.backbone} · seed {cfg.seed} · init={cfg.init} · aug={cfg.aug} · "
                f"loss={cfg.loss} · mix={cfg.mix} · img={cfg.img_size}")

    peak_mem = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else None
    summary = {
        "exp_id": cfg.exp_id, "seed": cfg.seed, "fold": cfg.fold, "backbone": cfg.backbone,
        "weight_tag": getattr(model, "weight_tag", None), "init": cfg.init, "drop_rate": cfg.drop_rate,
        "aug": cfg.aug, "mix": cfg.mix, "mix_alpha": cfg.mix_alpha, "loss": cfg.loss,
        "label_smoothing": cfg.label_smoothing, "focal_gamma": cfg.focal_gamma,
        "class_weight_beta": cfg.class_weight_beta, "sampler": cfg.sampler,
        "img_size": cfg.img_size, "epochs": cfg.epochs, "batch_size": cfg.batch_size,
        "lr_backbone": cfg.lr_backbone, "lr_head": cfg.lr_head, "weight_decay": cfg.weight_decay,
        "warmup_epochs": cfg.warmup_epochs, "ema_decay": cfg.ema_decay, "grad_clip": cfg.grad_clip,
        "amp": amp_on, "params_m": model_mod.count_params(model),
        "gmac": model_mod.count_gmacs(model, cfg.img_size), "gmac_tool": model_mod.GMAC_TOOL,
        "best_epoch": best["epoch"], "n_train": len(train_df), "n_val": len(val_df),
        "n_test": len(test_df), "val_top1": best["top1"], "val_macro_f1": best["macro_f1"],
        "val_loss": best.get("val_loss"), "val_balanced_acc": best.get("val_balanced_acc"),
        "val_ece": best.get("val_ece"), "val_nll": best.get("val_nll"),
        "train_seconds_per_epoch": float(np.mean([h["seconds"] for h in history])),
        "total_seconds": time.perf_counter() - t_start, "peak_gpu_mem_mb": peak_mem,
        "split": split_info, "test": test_info, "versions": _version_info(),
        "history": history,
    }
    (rdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), "utf-8")
    # done.json ghi CUỐI CÙNG: dấu hiệu "lần chạy này đã xong" để notebook cache-aware bỏ qua khi chạy lại
    (rdir / "done.json").write_text(json.dumps(
        {k: v for k, v in summary.items() if k != "history"}, indent=2, ensure_ascii=False), "utf-8")
    print(f"-> {cfg.exp_id} seed{cfg.seed}: best epoch {best['epoch']}, "
          f"val macro-F1 {best['macro_f1']:.4f}, val top-1 {best['top1']:.4f}, "
          f"{summary['total_seconds'] / 60:.1f} phút", flush=True)
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict, ép kiểu theo field của Config."""
    types = {f.name: f.type for f in fields(Config)}
    out: dict = {}
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"tham số phải có dạng KEY=VALUE, nhận: {item!r}")
        key, raw = item.split("=", 1)
        key, raw = key.strip(), raw.strip()
        if key not in types:
            raise KeyError(f"Config không có field {key!r}. Các field hợp lệ: {sorted(types)}")
        low = raw.lower()
        if low in ("none", "null", ""):
            out[key] = None
            continue
        tname = str(types[key])
        if "bool" in tname:
            if low not in ("true", "false", "1", "0", "yes", "no"):
                raise ValueError(f"{key}: cần true/false, nhận {raw!r}")
            out[key] = low in ("true", "1", "yes")
        elif "int" in tname and "float" not in tname:
            out[key] = int(raw)
        elif "float" in tname:
            out[key] = float(raw)
        else:
            out[key] = raw
    return out


def main() -> None:
    """Điểm vào dòng lệnh: `python train.py --set exp_id=B01 backbone=resnet50 seed=0`."""
    ap = argparse.ArgumentParser(description="Chạy một thí nghiệm Lab Day 2 (DeepWeeds)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="ghi đè field của Config, ví dụ: --set exp_id=B01 backbone=resnet50")
    ap.add_argument("--json-out", default=None, help="lưu dict tóm tắt ra file JSON")
    args = ap.parse_args()

    cfg = Config(**parse_overrides(args.set))
    summary = run(cfg)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {k: v for k, v in summary.items() if k != "history"}, indent=2, ensure_ascii=False), "utf-8")


if __name__ == "__main__":
    main()
