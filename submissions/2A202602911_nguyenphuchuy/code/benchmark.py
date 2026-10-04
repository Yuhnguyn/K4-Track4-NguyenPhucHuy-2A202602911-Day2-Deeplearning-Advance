"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

ĐÃ HOÀN THIỆN từ starter/.

Quy tắc đo (vi phạm bị trừ điểm, RUBRIC mục 3):
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() (hoặc CUDA event) TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch
  - chọn và ghi rõ có tính tiền xử lý hay không (ở đây: KHÔNG tính tiền xử lý ảnh, chỉ forward)
"""
from __future__ import annotations

import time

import numpy as np
import torch


def _sync_fn(device):
    """Hàm đồng bộ tương ứng với thiết bị: torch.cuda.synchronize trên GPU, None trên CPU."""
    if isinstance(device, torch.device):
        device = device.type
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.synchronize
    return None


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây.

    `sync` là hàm đồng bộ (ví dụ torch.cuda.synchronize) hoặc None trên CPU.
    Trả về {"p50", "p95", "p99", "mean", "min", "max", "n"} (đơn vị ms).
    """
    if iters < 1:
        raise ValueError("iters phải >= 1")
    for _ in range(int(warmup)):                      # warmup: bỏ những lần đầu (cuBLAS, cấp phát...)
        fn()
    if sync is not None:
        sync()

    times = np.empty(int(iters), dtype=np.float64)
    for i in range(int(iters)):
        if sync is not None:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        times[i] = (time.perf_counter() - t0) * 1000.0
    return {"p50": float(np.percentile(times, 50)), "p95": float(np.percentile(times, 95)),
            "p99": float(np.percentile(times, 99)), "mean": float(times.mean()),
            "min": float(times.min()), "max": float(times.max()), "n": int(iters)}


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    dtype: "fp32" | "amp" (autocast) | "fp16" (model.half()).
    Trả về dict ghi thẳng được vào sheet `Latency` của results.xlsx.
    """
    dtype = (dtype or "fp32").lower()
    if dtype not in ("fp32", "amp", "fp16"):
        raise ValueError(f"dtype không hợp lệ: {dtype} (chọn fp32 | amp | fp16)")
    dev = torch.device(device if (torch.cuda.is_available() or not str(device).startswith("cuda")) else "cpu")
    # Bản sao: dtype "fp16" gọi model.half() — không được làm hỏng model của người gọi.
    import copy as _copy
    model = _copy.deepcopy(model).to(dev).eval()

    x = torch.randn(batch_size, 3, img_size, img_size, device=dev)
    if dtype == "fp16":
        model = model.half()
        x = x.half()

    if dtype == "amp":
        try:
            ctx = torch.amp.autocast("cuda", dtype=torch.float16)
        except (AttributeError, TypeError):  # pragma: no cover
            ctx = torch.cuda.amp.autocast()
    else:
        ctx = torch.inference_mode()

    with torch.inference_mode():
        def fn():
            with ctx:
                model(x)

        res = bench(fn, warmup=warmup, iters=iters, sync=_sync_fn(dev))

    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"
    out = {"gpu": gpu, "dtype": dtype, "batch": int(batch_size), "img_size": int(img_size),
           "fused_bn": bool(getattr(model, "fused_pairs", 0) > 0),
           "p50": res["p50"], "p95": res["p95"], "p99": res["p99"],
           "mean": res["mean"], "min": res["min"], "n": res["n"],
           "images_per_s": float(batch_size) / (res["p50"] / 1000.0),
           "throughput_per_s": float(batch_size) / (res["p50"] / 1000.0),
           "torch": torch.__version__}
    return out


def tta_latency(model, k_views: int, **kw) -> dict:
    """Độ trễ của TTA K view: đo THẬT (forward K lần trong cùng một lần đo, như lúc triển khai)
    và so với xấp xỉ K * p50 của một view.

    Các tham số còn lại (`batch_size`, `img_size`, `dtype`, `device`, `warmup`, `iters`) giống
    `latency_report`.
    """
    if k_views < 1:
        raise ValueError("k_views phải >= 1")
    device = kw.get("device", "cuda")
    batch_size = int(kw.get("batch_size", 1))
    img_size = int(kw.get("img_size", 224))
    dtype = str(kw.get("dtype", "fp32")).lower()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    import copy as _copy
    model = _copy.deepcopy(model).to(dev).eval()   # bản sao: "fp16" gọi model.half()

    x = torch.randn(batch_size, 3, img_size, img_size, device=dev)
    if dtype == "fp16":
        model = model.half()
        x = x.half()
    if dtype == "amp":
        try:
            ctx = torch.amp.autocast("cuda", dtype=torch.float16)
        except (AttributeError, TypeError):  # pragma: no cover
            ctx = torch.cuda.amp.autocast()
    else:
        ctx = torch.inference_mode()

    with torch.inference_mode():
        def one():
            with ctx:
                model(x)

        def k_times():
            with ctx:
                for _ in range(int(k_views)):
                    model(x)

        single = bench(one, warmup=kw.get("warmup", 10), iters=kw.get("iters", 100),
                       sync=_sync_fn(dev))
        multi = bench(k_times, warmup=kw.get("warmup", 10), iters=kw.get("iters", 100),
                      sync=_sync_fn(dev))

    single["views"] = 1
    multi["views"] = int(k_views)
    multi["predicted_k_times_p50"] = single["p50"] * int(k_views)
    multi["ratio_vs_k_times"] = multi["p50"] / max(single["p50"] * int(k_views), 1e-9)
    return {"single_view": single, "k_views": multi}
