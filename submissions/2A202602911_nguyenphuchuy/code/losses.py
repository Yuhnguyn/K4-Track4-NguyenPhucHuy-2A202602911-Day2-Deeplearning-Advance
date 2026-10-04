"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

ĐÃ HOÀN THIỆN từ starter/. Giao diện giữ nguyên:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar

Liên hệ slide Day 2: label smoothing (trang 56), focal loss (trang 57), Mixup/CutMix (trang 48).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Cross-entropy / label smoothing / focal
# --------------------------------------------------------------------------- #
class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K.

    Cách cài: dùng thẳng `torch.nn.functional.cross_entropy(..., label_smoothing=eps)` của PyTorch
    (đã kiểm chứng: eps = 0 cho lại đúng CE tới < 1e-6, xem tests/test_code.py). Tự khai triển lại
    công thức cho kết quả tương đương; chọn cách của PyTorch để tránh sai số làm tròn.
    """

    def __init__(self, smoothing: float = 0.1, weight: torch.Tensor | None = None,
                 reduction: str = "mean"):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing phải trong [0, 1), nhận {smoothing}")
        self.smoothing = float(smoothing)
        self.reduction = reduction
        self.register_buffer("weight", None if weight is None else weight.detach().float())

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target, weight=self.weight,
                               label_smoothing=self.smoothing, reduction=self.reduction)


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)  (slide trang 57).

    gamma = 0 phải cho lại ĐÚNG cross-entropy (đã có test, sai số < 1e-6).
    `alpha`: None hoặc vector trọng số theo lớp (độ dài K) hoặc scalar.
    """

    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        if alpha is None:
            self.register_buffer("alpha", None)
        elif isinstance(alpha, (int, float)):
            self.register_buffer("alpha", torch.tensor(float(alpha)))
        else:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        logpt = logp.gather(1, target.view(-1, 1)).squeeze(1)      # log p_t
        pt = logpt.exp()
        loss = -((1.0 - pt) ** self.gamma) * logpt                 # gamma=0 -> -log p_t = CE
        if self.alpha is not None:
            a = self.alpha if self.alpha.ndim == 0 else self.alpha.to(logits.device)[target]
            loss = loss * a
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def class_weights(counts, beta: float = 0.0) -> torch.Tensor:
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN.

    - beta = 0: w_c = (1 / n_c) chuẩn hoá về trung bình 1
    - beta > 0: class-balanced theo "số mẫu hiệu dụng" w_c = (1 - beta) / (1 - beta ** n_c)
      (Cui et al. arXiv:1901.05555), chuẩn hoá tổng về đúng số lớp
    Chỉ dùng số liệu của train, không dùng val hay test.
    """
    n = np.asarray(counts, dtype=np.float64).ravel()
    if (n <= 0).any():
        raise ValueError("counts phải là số ảnh dương cho mọi lớp")
    if beta > 0:
        w = (1.0 - beta) / (1.0 - np.power(beta, n))
        w = w / w.sum() * len(w)
    else:
        w = 1.0 / n
        w = w / w.mean()
    return torch.as_tensor(w, dtype=torch.float32)


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: "ce", "ls" (label smoothing), "focal", "ce_weighted".

    kw dùng được: smoothing, gamma, weight (tensor K cho các lớp), alpha (focal), reduction.
    """
    kind = (kind or "ce").lower()
    smoothing = float(kw.get("smoothing", kw.get("label_smoothing", 0.0)) or 0.0)
    gamma = float(kw.get("gamma", kw.get("focal_gamma", 2.0)) or 0.0)
    weight = kw.get("weight", None)
    alpha = kw.get("alpha", None)
    reduction = kw.get("reduction", "mean")

    if weight is not None:
        weight = torch.as_tensor(weight, dtype=torch.float32)

    if kind == "ce":
        return nn.CrossEntropyLoss(weight=weight, reduction=reduction)
    if kind in ("ls", "label_smoothing"):
        return LabelSmoothingCE(smoothing=smoothing if smoothing > 0 else 0.1, weight=weight,
                                reduction=reduction)
    if kind == "focal":
        return FocalLoss(gamma=gamma, alpha=alpha if alpha is not None else weight,
                         reduction=reduction)
    if kind == "ce_weighted":
        if weight is None:
            raise ValueError("kind='ce_weighted' cần weight (dùng losses.class_weights)")
        return nn.CrossEntropyLoss(weight=weight, reduction=reduction)
    raise ValueError(f"loss không hợp lệ: {kind} (chọn ce | ls | focal | ce_weighted)")


# --------------------------------------------------------------------------- #
# Mixup / CutMix
# --------------------------------------------------------------------------- #
def mix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn một batch ảnh và nhãn.

    - lam ~ Beta(alpha, alpha)
    - mode="mixup": x_mix = lam * x + (1 - lam) * x[perm]
    - mode="cutmix": cắt hộp chữ nhật từ x[perm] dán vào x; lam được điều chỉnh theo DIỆN TÍCH
      THỰC của hộp sau khi cắt ra ngoài biên (slide trang 48): lam = 1 - area_dán / (H * W)
    Trả về (x_mix, (y_a, y_b, lam)) với y_a = y, y_b = y[perm].
    """
    mode = (mode or "cutmix").lower()
    if x.size(0) < 2:
        raise ValueError("mix_batch cần batch >= 2 ảnh")
    perm = torch.randperm(x.size(0), device=x.device)
    lam = float(np.random.beta(alpha, alpha))

    if mode == "mixup":
        x_mix = lam * x + (1.0 - lam) * x[perm]
    elif mode == "cutmix":
        h, w = x.shape[-2:]
        ratio = float(np.sqrt(1.0 - lam))
        cut_w, cut_h = int(w * ratio), int(h * ratio)
        cx = int(np.random.randint(w))
        cy = int(np.random.randint(h))
        x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)
        y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
        x_mix = x.clone()
        if x2 > x1 and y2 > y1:
            x_mix[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
            lam = 1.0 - ((x2 - x1) * (y2 - y1) / (w * h))   # diện tích THỰC đã dán
    else:
        raise ValueError(f"mode không hợp lệ: {mode} (chọn mixup | cutmix)")

    return x_mix, (y, y[perm], lam)


def mixed_loss(criterion, logits: torch.Tensor, targets) -> torch.Tensor:
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b).

    Lưu ý: accuracy trên batch đã trộn không còn nghĩa bình thường; đánh giá bằng val.
    """
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
