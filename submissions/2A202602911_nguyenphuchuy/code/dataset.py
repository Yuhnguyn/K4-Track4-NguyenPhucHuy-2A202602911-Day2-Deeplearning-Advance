"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

ĐÃ HOÀN THIỆN từ starter/. Giữ nguyên tên hàm, tham số và kiểu vào/ra của khung;
các tham số thêm vào đều có giá trị mặc định nên không phá vỡ giao diện.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1.
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Tổng số ảnh của DeepWeeds (không đổi giữa các fold, mỗi fold là một cách chia của cùng 17.509 ảnh).
EXPECTED_TOTAL = 17509
# Ngưỡng lệch tỉ lệ 60/20/20 được phép trước khi phải báo giảng viên (README.md mục 2.1).
RATIO_TOL = 0.01


# --------------------------------------------------------------------------- #
# 1. Đọc và kiểm tra chia dữ liệu
# --------------------------------------------------------------------------- #
def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Trả về ba DataFrame cột `Filename, Label` theo đúng thứ tự trong file CSV
    (KHÔNG sửa, lọc, xáo trộn hay chia lại dữ liệu).
    """
    labels_dir = Path(labels_dir)
    frames = []
    for part in ("train", "val", "test"):
        path = labels_dir / f"{part}_subset{fold}.csv"
        if not path.exists():
            raise FileNotFoundError(f"thiếu file chia dữ liệu: {path} (S1: phải dùng file chia sẵn của tác giả)")
        df = pd.read_csv(path)
        missing = [c for c in ("Filename", "Label") if c not in df.columns]
        if missing:
            raise ValueError(f"{path}: thiếu cột {missing}")
        df = df[["Filename", "Label"]].copy()
        df["Filename"] = df["Filename"].astype(str)
        df["Label"] = df["Label"].astype(int)
        if df["Filename"].duplicated().any():
            raise ValueError(f"{path}: có Filename bị trùng")
        bad = df.loc[~df["Label"].between(0, NUM_CLASSES - 1), "Label"]
        if len(bad):
            raise ValueError(f"{path}: nhãn ngoài khoảng 0..{NUM_CLASSES - 1}: {sorted(set(bad))}")
        frames.append(df)
    return tuple(frames)  # type: ignore[return-value]


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path | None = None, expected_total: int = EXPECTED_TOTAL,
                verbose: bool = True) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). Raise ngay nếu vi phạm.

    Kiểm tra:
      1. số ảnh mỗi tập + số ảnh mỗi lớp trong từng tập (kỳ vọng xấp xỉ 60/20/20)
      2. giao của từng cặp tập theo Filename phải RỖNG
      3. hợp ba tập phải bằng đúng `expected_total` ảnh (mặc định 17.509)
      4. mọi Filename đều tồn tại trong `images_dir`
    Trả về dict số liệu để dán vào báo cáo.
    """
    parts = {"train": train_df, "val": val_df, "test": test_df}
    names = {k: set(v["Filename"]) for k, v in parts.items()}
    n = {k: int(len(v)) for k, v in parts.items()}
    per_class = {k: v["Label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0).astype(int).tolist()
                 for k, v in parts.items()}

    # (2) giao rỗng
    overlap = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap[f"{a}&{b}"] = len(names[a] & names[b])
    bad = {k: v for k, v in overlap.items() if v}
    if bad:
        raise AssertionError(f"giao giữa các tập KHÁC RỖNG (rò rỉ dữ liệu): {bad}")

    # (3) hợp đủ tổng số ảnh
    union = names["train"] | names["val"] | names["test"]
    if len(union) != expected_total:
        raise AssertionError(f"hợp ba tập = {len(union)} ảnh, kỳ vọng {expected_total}")
    if sum(n.values()) != len(union):
        raise AssertionError(f"tổng số dòng {sum(n.values())} != số ảnh hợp {len(union)} (có Filename xuất hiện 2 lần)")

    # (1) tỉ lệ 60/20/20 - chỉ cảnh báo, README yêu cầu báo giảng viên chứ không tự sửa
    ratio = {k: v / len(union) for k, v in n.items()}
    ratio_warnings = [f"{k}: {100 * ratio[k]:.2f}% (kỳ vọng {100 * target:.0f}%)"
                      for k, target in (("train", 0.6), ("val", 0.2), ("test", 0.2))
                      if abs(ratio[k] - target) > RATIO_TOL]

    # (4) file ảnh tồn tại
    missing_files = []
    if images_dir is not None:
        images_dir = Path(images_dir)
        if not images_dir.is_dir():
            raise FileNotFoundError(f"thư mục ảnh không tồn tại: {images_dir}")
        present = set(os.listdir(images_dir))
        miss = sorted(f for s in names.values() for f in s if f not in present)
        if miss:
            raise FileNotFoundError(f"{len(miss)} ảnh trong CSV không có trong {images_dir}, "
                                    f"ví dụ: {miss[:5]}")
        missing_files = miss

    info = {"n": n, "ratio": ratio, "per_class": per_class, "overlap": overlap,
            "total": len(union), "images_dir": str(images_dir) if images_dir else None,
            "ratio_warnings": ratio_warnings, "missing_files": missing_files}

    if verbose:
        print(f"Tổng ảnh: {len(union)}  |  train {n['train']} / val {n['val']} / test {n['test']}")
        print(f"Tỉ lệ  : {ratio['train']:.4f} / {ratio['val']:.4f} / {ratio['test']:.4f}")
        print(f"Giao   : " + ", ".join(f"{k}={v}" for k, v in overlap.items()))
        print(f"{'Lớp':<16}{'train':>8}{'val':>8}{'test':>8}{'tổng':>8}")
        for i, cname in enumerate(CLASS_NAMES):
            row = [per_class[p][i] for p in ("train", "val", "test")]
            print(f"{cname:<16}{row[0]:>8}{row[1]:>8}{row[2]:>8}{sum(row):>8}")
        if ratio_warnings:
            print("CẢNH BÁO tỉ lệ:", "; ".join(ratio_warnings), "-> báo giảng viên trước khi chạy tiếp")
    return info


# --------------------------------------------------------------------------- #
# 2. Transform / augmentation
# --------------------------------------------------------------------------- #
# Trục B (GUIDE.md mục 3): ghép bằng dấu "+", ví dụ "color", "trivial", "color+randaug".
_PIL_AUGS = {
    "color": lambda: transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    "gray": lambda: transforms.RandomGrayscale(p=0.1),
    "blur": lambda: transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    "trivial": lambda: transforms.TrivialAugmentWide(),
    "randaug": lambda: transforms.RandAugment(num_ops=2, magnitude=9),
    "persp": lambda: transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
}
_TENSOR_AUGS = {
    "erase": lambda: transforms.RandomErasing(p=0.25, value="random"),
}
# Lật dọc KHÔNG được bật mặc định: ảnh cỏ dại chụp trên cao, lật dọc tạo ảnh "treo ngược"
# mà mô hình sẽ gặp lại ở thời gian thực -> chỉ thử như một thí nghiệm (trục B), không dùng ở công thức nền.
AUG_TOKENS = tuple(_PIL_AUGS) + tuple(_TENSOR_AUGS)


def eval_resize(img_size: int) -> int:
    """Cạnh ngắn dùng lúc đánh giá, giữ đúng tỉ lệ crop:resize như lúc train (256:224)."""
    return int(round(img_size * 256 / 224))


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic",
                     mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Tạo transform. `aug` chọn mức augmentation (trục B của GUIDE.md mục 3).

    Train (basic): RandomResizedCrop(img_size) + lật ngang + ToTensor + Normalize.
    Val/test: Resize(round(img_size*256/224)) + CenterCrop(img_size) + ToTensor + Normalize —
      cùng tỉ lệ crop:resize như lúc train, KHÔNG augmentation ngẫu nhiên.
    (`aug` bị bỏ qua khi train=False; muốn đổi tiền xử lý lúc đánh giá — ví dụ dò độ phân giải
     ở Bước 3 — thì gọi hàm này với `img_size` khác.)

    `aug` = "none" | "basic" | các token trong AUG_TOKENS ghép bằng "+" (color, gray, blur,
    trivial, randaug, persp, erase). Ví dụ: "basic+color", "color+randaug+erase".
    """
    to_tensor_norm = [transforms.ToTensor(), transforms.Normalize(list(mean), list(std))]

    if not train:
        return transforms.Compose([
            transforms.Resize(eval_resize(img_size), antialias=True),
            transforms.CenterCrop(img_size),
            *to_tensor_norm,
        ])

    tokens = [] if aug in ("none", None, "") else str(aug).split("+")
    unknown = [t for t in tokens if t not in _PIL_AUGS and t not in _TENSOR_AUGS and t != "basic"]
    if unknown:
        raise ValueError(f"aug không hợp lệ: {unknown}; chọn trong {('basic', 'none', *AUG_TOKENS)}")

    ops = [transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.75, 1.3333),
                                        antialias=True),
           transforms.RandomHorizontalFlip(p=0.5)]
    tensor_ops = []
    for t in tokens:
        if t in _PIL_AUGS:
            ops.append(_PIL_AUGS[t]())
        elif t in _TENSOR_AUGS:
            tensor_ops.append(_TENSOR_AUGS[t]())
    return transforms.Compose([*ops, *to_tensor_norm, *tensor_ops])


# --------------------------------------------------------------------------- #
# 3. Dataset / DataLoader
# --------------------------------------------------------------------------- #
class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    __getitem__(i) trả về (ảnh đã transform, nhãn int, tên file str).
    Tên file cần có để ghi `predictions/*.csv` đúng định dạng của eval.py.
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.filenames = df["Filename"].astype(str).tolist()
        self.labels = df["Label"].astype(int).tolist()
        self.images_dir = str(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int):
        name = self.filenames[i]
        with Image.open(os.path.join(self.images_dir, name)) as im:
            img = im.convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
        return img, self.labels[i], name


def _worker_init_fn(seed: int):
    """Seed cho worker của DataLoader để augmentation ngẫu nhiên tái lập được."""
    def _init(worker_id: int) -> None:
        s = seed + worker_id
        random.seed(s)
        np.random.seed(s % (2 ** 32))
    return _init


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2,
                seed: int = 0, drop_last: bool | None = None, pin_memory: bool = True):
    """Tạo DataLoader.

    - train=True: shuffle (hoặc `sampler="balanced"` -> WeightedRandomSampler 1/n_c, trục D).
      train=False: KHÔNG shuffle, giữ nguyên thứ tự df để ghép logit với Filename.
    - drop_last: mặc định = train (batch cuối quá nhỏ làm BatchNorm không ổn định).
    - worker_init_fn seed theo `seed` để augmentation tái lập.
    """
    ds = DeepWeedsDataset(df, images_dir, transform)
    kwargs: dict = {}
    if train and sampler == "balanced":
        counts = df["Label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0).to_numpy(dtype=np.float64)
        per_sample = 1.0 / np.maximum(counts, 1.0)
        weights = torch.as_tensor(per_sample[df["Label"].to_numpy()], dtype=torch.double)
        gen = torch.Generator()
        gen.manual_seed(seed)
        kwargs["sampler"] = WeightedRandomSampler(weights, num_samples=len(df), replacement=True,
                                                 generator=gen)
    elif train and sampler not in (None, "none"):
        raise ValueError(f"sampler không hợp lệ: {sampler} (chọn None hoặc 'balanced')")
    elif train:
        kwargs["shuffle"] = True

    if drop_last is None:
        drop_last = bool(train)

    cuda = torch.cuda.is_available()
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      pin_memory=pin_memory and cuda, drop_last=drop_last,
                      worker_init_fn=_worker_init_fn(seed), **kwargs)
