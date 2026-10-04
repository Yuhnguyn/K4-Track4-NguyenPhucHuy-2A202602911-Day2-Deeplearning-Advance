"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

ĐÃ HOÀN THIỆN từ starter/. Giao diện giữ nguyên:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)

Mọi hàm chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val;
nhiệt độ T khớp trên VAL rồi áp dụng sang test (README.md, S2 và S4).

Ghi chú giới hạn (đã đo ở Bước 3):
  - `views_multiscale` và `views_multicrop` đổi kích thước/độ phân giải đầu vào: chỉ dùng được với
    CNN có global pooling (ResNet/ResNeXt/ConvNeXt/EfficientNet/MobileNet). ViT/Swin/DeiT cần lưới
    patch cố định (và vị trí cửa sổ) -> forward sẽ lỗi hoặc cho kết quả vô nghĩa; Bước 3 ghi rõ điều này.
  - `fuse_conv_bn` chỉ áp dụng cho kiến trúc có BatchNorm. ViT/Swin/ConvNeXt dùng LayerNorm -> không gộp.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Chạy model và gom logit
# --------------------------------------------------------------------------- #
def _autocast(enabled: bool):
    """autocast tương thích cả torch >= 2.4 và torch cũ (Colab hay đổi API amp)."""
    try:
        return torch.amp.autocast("cuda", enabled=bool(enabled))
    except (AttributeError, TypeError):  # pragma: no cover
        return torch.cuda.amp.autocast(enabled=bool(enabled))


@torch.inference_mode()
def predict_logits(model, loader, device, view=None):
    """Chạy model trên loader và gom logit theo đúng thứ tự file.

    `view` là hàm biến đổi batch ảnh trước khi đưa vào model (ví dụ `view_hflip`), hoặc None.
    Nếu `view` trả về MỘT tensor -> logits có dạng (N, K).
    Nếu `view` trả về list/tuple nhiều tensor (TTA nhiều view) -> logits có dạng (V, N, K),
    đưa thẳng vào `aggregate_views`.
    """
    model.eval()
    names: list[str] = []
    ys: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    amp = torch.cuda.is_available()
    for x, y, fname in loader:
        x = x.to(device, non_blocking=True)
        views = view(x) if view is not None else x
        if not isinstance(views, (list, tuple)):
            views = [views]
        with _autocast(amp):
            out = [model(v.to(device)).detach().float().cpu().numpy() for v in views]
        zs.append(out[0] if len(out) == 1 else np.stack(out))   # (N,K) hoặc (V,N,K)
        ys.append(y.numpy())
        names.extend(list(fname))
    y_true = np.concatenate(ys) if ys else np.zeros(0, dtype=np.int64)
    if not zs:
        return names, y_true, np.zeros((0, 9), dtype=np.float32)
    logits = zs[0] if len(zs) == 1 else np.concatenate(zs, axis=-2)
    return names, y_true, logits


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W)."""
    return torch.flip(x, dims=[3])


def views_multicrop(x, crop: int, flip: bool = True):
    """5 crop (4 góc + giữa) kích thước `crop`, kèm bản lật nếu `flip=True`.

    Dùng cho CNN: đưa ảnh ở độ phân giải LỚN HƠN (index 256) rồi cắt 5 vị trí cỡ 224 — mỗi crop
    đúng cỡ lúc train. Trả về list các batch.
    """
    _, _, h, w = x.shape
    if crop > min(h, w):
        raise ValueError(f"crop={crop} lớn hơn kích thước ảnh {h}x{w}")
    boxes = [(0, 0), (0, w - crop), (h - crop, 0), (h - crop, w - crop),
             ((h - crop) // 2, (w - crop) // 2)]
    views = []
    for top, left in boxes:
        v = x[:, :, top:top + crop, left:left + crop]
        views.append(v)
        if flip:
            views.append(torch.flip(v, dims=[3]))
    return views


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes`, trả về list các batch (dò độ phân giải, FixRes)."""
    return [F.interpolate(x, size=(int(s), int(s)), mode="bilinear", align_corners=False,
                          antialias=True) for s in sizes]


# --------------------------------------------------------------------------- #
# Gộp view / ensemble
# --------------------------------------------------------------------------- #
def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K lượt chạy của TTA thành một dự đoán (slide trang 62).

      - space="prob":  trung bình softmax của từng view
      - space="logit": trung bình logit rồi softmax
    Trả về xác suất (N, K) đã chuẩn hoá.
    """
    z = np.asarray(logits_per_view, dtype=np.float64)
    if z.ndim == 2:
        z = z[None, ...]
    if z.ndim != 3:
        raise ValueError(f"logits_per_view phải có dạng (V, N, C) hoặc (N, C), nhận {z.shape}")
    space = (space or "prob").lower()
    if space == "prob":
        p = _softmax_np(z)
        probs = p.mean(axis=0)
    elif space == "logit":
        probs = _softmax_np(z.mean(axis=0))
    else:
        raise ValueError(f"space không hợp lệ: {space} (chọn prob | logit)")
    return probs / probs.sum(axis=1, keepdims=True)


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (khác backbone hoặc khác seed).

    Chỉ ghép các mô hình trên CÙNG tập ảnh và cùng thứ tự file.
    """
    arrs = [np.asarray(p, dtype=np.float64) for p in list_of_probs]
    shapes = {a.shape for a in arrs}
    if len(shapes) != 1:
        raise ValueError(f"các mảng xác suất phải cùng shape, nhận {sorted(shapes)}")
    p = np.mean(arrs, axis=0)
    return p / p.sum(axis=1, keepdims=True)


def _softmax_np(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# --------------------------------------------------------------------------- #
# Temperature scaling
# --------------------------------------------------------------------------- #
def _nll(logits: np.ndarray, labels: np.ndarray, T: float) -> float:
    p = _softmax_np(np.asarray(logits, dtype=np.float64) / T)
    return float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, None)).mean())


def fit_temperature(val_logits, val_labels) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T)  (slide trang 69).

    Tối ưu một tham số bằng LBFGS trên log T (số float64, không cần GPU); nếu không hội tụ thì
    dò lưới thô rồi tinh. Accuracy KHÔNG đổi vì thứ tự lớp không đổi. KHÔNG khớp T trên test.
    """
    logits = np.asarray(val_logits, dtype=np.float64)
    labels = np.asarray(val_labels, dtype=np.int64)
    if len(logits) != len(labels):
        raise ValueError("val_logits và val_labels phải cùng độ dài")

    best_T, best_nll = 1.0, _nll(logits, labels, 1.0)
    try:
        z = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([z], lr=0.1, max_iter=200)

        def closure():
            opt.zero_grad()
            p = torch.softmax(torch.as_tensor(logits, dtype=torch.float64) / z.exp(), dim=1)
            loss = F.nll_loss(torch.log(p.clamp_min(1e-12)),
                              torch.as_tensor(labels, dtype=torch.long))
            loss.backward()
            return loss

        opt.step(closure)
        T = float(z.exp().item())
        if np.isfinite(T) and 1e-3 < T < 1e3 and _nll(logits, labels, T) < best_nll:
            best_T, best_nll = T, _nll(logits, labels, T)
    except Exception:  # pragma: no cover - LBFGS hiếm khi lỗi, lưới là đường dự phòng
        pass

    grid = np.concatenate([np.linspace(0.5, 2.0, 31), np.linspace(2.1, 6.0, 40)])
    for T in grid:
        v = _nll(logits, labels, float(T))
        if v < best_nll:
            best_T, best_nll = float(T), v
    return float(best_T)


def apply_temperature(logits, T: float):
    """Trả về softmax(logits / T)."""
    if not (T > 0):
        raise ValueError(f"T phải > 0, nhận {T}")
    return _softmax_np(np.asarray(logits, dtype=np.float64) / float(T))


# --------------------------------------------------------------------------- #
# Gộp BatchNorm vào conv
# --------------------------------------------------------------------------- #
def _fuse_pair(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    if bn.running_mean is None or bn.running_var is None:
        raise ValueError("BatchNorm chưa có running stats: chạy ít nhất 1 forward ở chế độ train trước khi gộp")
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                      conv.padding, conv.dilation, conv.groups, bias=True,
                      padding_mode=conv.padding_mode)
    w = conv.weight.detach().clone()
    b = (conv.bias.detach().clone() if conv.bias is not None
         else torch.zeros(conv.out_channels, device=w.device, dtype=w.dtype))
    mean = bn.running_mean.detach()
    std = torch.sqrt(bn.running_var.detach() + bn.eps)
    gamma = bn.weight.detach() if bn.weight is not None else torch.ones_like(mean)
    beta = bn.bias.detach() if bn.bias is not None else torch.zeros_like(mean)
    scale = gamma / std
    fused.weight.data = w * scale.reshape(-1, 1, 1, 1)
    fused.bias.data = (b - mean) * scale + beta
    return fused.to(w.device, w.dtype)


def _all_modules(m):
    yield m
    for _, c in m.named_children():
        yield from _all_modules(c)


def fuse_conv_bn(model):
    """Gộp BatchNorm vào tích chập liền trước (slide trang 71, 75):
        w' = gamma * w / sqrt(var + eps)        b' = beta + gamma * (b - mean) / sqrt(var + eps)
    model.eval() trước; mỗi cặp (Conv2d, BatchNorm2d) liền kề -> conv mới có bias, BN thành Identity.
    Số cặp đã gộp lưu ở thuộc tính `model.fused_pairs` (0 với kiến trúc không có BN: ViT/Swin/ConvNeXt).
    """
    model.eval()
    n = 0
    for m in list(_all_modules(model)):
        kids = list(m.named_children())
        for i in range(len(kids) - 1):
            na, a = kids[i]
            nb, b = kids[i + 1]
            if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
                setattr(m, na, _fuse_pair(a, b))
                setattr(m, nb, nn.Identity())
                n += 1
    model.fused_pairs = n
    return model


@torch.inference_mode()
def fuse_error(model, x) -> float:
    """Sai số tuyệt đối lớn nhất trước/sau khi gộp BN (kỳ vọng <= ~1e-5). Dùng làm bằng chứng."""
    import copy
    before = copy.deepcopy(model).eval()
    y0 = before(x)
    y1 = model(x)
    return float((y0 - y1).abs().max())
