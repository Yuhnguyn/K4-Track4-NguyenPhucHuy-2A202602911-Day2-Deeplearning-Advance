"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

ĐÃ HOÀN THIỆN từ starter/. Giao diện giữ nguyên:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

import torch

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",      # hoặc vit_small_patch16_224
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100",      # mạng nhẹ
}

# Công cụ đã dùng để đếm GMAC ở lần gọi gần nhất (ghi vào results.xlsx để tái lập được).
GMAC_TOOL = "chưa gọi"


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Tạo model phân loại 9 lớp.

    `init` (trục A của GUIDE.md mục 3):
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ

    Trả về nn.Module của timm; gắn thêm 2 thuộc tính để ghi log:
      model.weight_tag   - tag trọng số tiền huấn luyện thực sự được tải
      model.pretrained_ok - có thực sự tải trọng số ImageNet hay không
    """
    import timm

    if init not in ("scratch", "frozen", "finetune"):
        raise ValueError(f"init không hợp lệ: {init} (chọn scratch | frozen | finetune)")
    use_pretrained = bool(pretrained) and init != "scratch"

    model = timm.create_model(name, pretrained=use_pretrained,
                              num_classes=num_classes, drop_rate=drop_rate)

    cfg = getattr(model, "pretrained_cfg", None) or {}
    tag = cfg.get("tag") or cfg.get("hf_hub_id") or cfg.get("architecture")
    if not use_pretrained:
        tag = "scratch (không tải trọng số)"
    model.weight_tag = str(tag) if tag else "unknown"
    model.pretrained_ok = use_pretrained
    model.init_mode = init

    if init == "frozen":
        freeze_backbone(model)
    return model


def head_param_ids(model) -> set[int]:
    """id() của các tham số thuộc head phân loại (mọi backbone của timm đều có get_classifier())."""
    try:
        head = model.get_classifier()
    except AttributeError as e:  # pragma: no cover
        raise AttributeError("model không có get_classifier(); không tách được nhóm head") from e
    return {id(p) for p in head.parameters()}


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head.

    Lưu ý (GUIDE.md mục 3.2): backbone đóng băng thì BatchNorm của backbone cũng phải ở chế độ
    eval — nếu không, running_mean/var bị cập nhật theo batch và kết quả khó hiểu.
    Việc đó do `keep_backbone_bn_eval(model)` làm, gọi lại sau mỗi `model.train()`.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.get_classifier().parameters():
        p.requires_grad_(True)
    return None


def keep_backbone_bn_eval(model) -> None:
    """Đưa mọi BatchNorm về eval (dùng sau model.train() khi init == 'frozen').

    Head phân loại của mọi backbone trong danh sách gợi ý là Linear (không có BN), nên đặt
    toàn bộ BN của model về eval là đúng: chỉ thống kê của backbone bị "đóng băng".
    """
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Chia tham số thành các nhóm như slide Day 2 trang 52.

    - backbone, ndim > 1        : lr = lr_backbone, weight_decay = weight_decay
    - backbone, ndim <= 1       : lr = lr_backbone, weight_decay = 0   (norm + bias)
    - head, ndim > 1            : lr = lr_head (thường gấp 10x), weight_decay = weight_decay
    - head, ndim <= 1           : lr = lr_head, weight_decay = 0       (bias của head)

    Bỏ qua tham số requires_grad == False. Trả về list[dict] đã sẵn sàng cho AdamW
    (nhóm rỗng bị loại để optimizer không báo lỗi).
    """
    head_ids = head_param_ids(model)
    buckets: dict[tuple, list] = {
        ("backbone", "decay"): [], ("backbone", "no_decay"): [],
        ("head", "decay"): [], ("head", "no_decay"): [],
    }
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        where = "head" if id(p) in head_ids else "backbone"
        which = "decay" if p.ndim > 1 else "no_decay"
        buckets[(where, which)].append(p)

    groups = []
    for (where, which), params in buckets.items():
        if not params:
            continue
        groups.append({
            "params": params,
            "lr": lr_head if where == "head" else lr_backbone,
            "weight_decay": 0.0 if which == "no_decay" else float(weight_decay),
            "name": f"{where}_{which}",
        })
    return groups


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    return float(sum(p.numel() for p in model.parameters()) / 1e6)


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size (slide tính MAC, không phải FLOPs 2x).

    Thử fvcore -> ptflops -> thop; ghi tên công cụ vào `model.GMAC_TOOL` (module-level).
    fvcore đếm FLOPs (1 MAC = 2 FLOP) nên chia 2. Số có thể lệch vài phần trăm giữa các công cụ.
    Đầu vào đặt cùng thiết bị với model (tránh lỗi device mismatch khi model đang ở CUDA).
    Trả về NaN nếu không có công cụ nào chạy được (ghi rõ trong báo cáo).
    """
    global GMAC_TOOL
    dev = next(model.parameters()).device
    x = torch.randn(1, 3, img_size, img_size, device=dev)
    was_training = model.training
    model.eval()

    def _try(desc, fn, scale):
        global GMAC_TOOL
        try:
            with torch.no_grad():
                value = fn()
            GMAC_TOOL = desc
            return float(value) * scale
        except ImportError:
            return None
        except Exception as exc:  # công cụ có thể không hỗ trợ kiến trúc/thiết bị này
            print(f"[count_gmacs] {desc} lỗi: {type(exc).__name__}: {exc}")
            return None

    try:
        v = _try("fvcore.nn.FlopCountAnalysis (FLOPs/2)",
                 lambda: __import__("fvcore.nn", fromlist=["FlopCountAnalysis"])
                 .FlopCountAnalysis(model, x).total(), 1 / 2 / 1e9)
        if v is not None:
            return v
        v = _try("ptflops.get_model_complexity_info",
                 lambda: __import__("ptflops", fromlist=["get_model_complexity_info"])
                 .get_model_complexity_info(model, (3, img_size, img_size), as_strings=False,
                                            print_per_layer_stat=False, verbose=False)[0], 1 / 1e9)
        if v is not None:
            return v
        v = _try("thop.profile",
                 lambda: __import__("thop", fromlist=["profile"]).profile(model, inputs=(x,),
                                                                         verbose=False)[0], 1 / 1e9)
        if v is not None:
            return v
        GMAC_TOOL = "KHÔNG CÓ công cụ đếm chạy được (fvcore/ptflops/thop)"
        return float("nan")
    finally:
        model.train(was_training)
