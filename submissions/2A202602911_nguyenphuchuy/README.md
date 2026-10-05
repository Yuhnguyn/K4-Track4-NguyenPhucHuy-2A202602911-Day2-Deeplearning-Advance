# Submission — Lab Day 2 (Track 4) · Nguyễn Phúc Huy · 2A202602911

## Kết quả test (chính thức, tính bằng `eval.py` của đề)

| | Mốc `T00` + `I00` | **Chung kết `F01`** |
|---|---|---|
| macro-F1 (mean ± std, 3 seed) | 0.7977 ± 0.0008 | **0.9734 ± 0.0015** |
| top-1 | 80.83 % | **97.93 %** |
| ECE sau hiệu chuẩn | 0.0217 | **0.0050** |

Δ = **+0.1758** macro-F1. Tự chấm RUBRIC phần I: **20/20**.
Cấu hình chung kết: **convnext_tiny** + tổ hợp công thức `T12` + suy luận `I04_256` (1 view @256) + temperature scaling (T = 1.5060).
Độ trễ batch-1 fp32: p50 5.56 ms · p95 5.82 ms (ngân sách 100 ms).

## Chạy lại

**Cách 1 — Kaggle (phiên chạy đầy đủ, xem được output từng cell):**

```
https://www.kaggle.com/code/nguynhuy9669/lab-day2-track4-2a202602911-r2
Settings: Accelerator = GPU (T4) · Internet = On   ->   Run all
```

**Cách 2 — Colab:** mở `code/lab_day2.ipynb` từ GitHub rồi *Run all* (Runtime → T4 GPU):

```
https://colab.research.google.com/github/Yuhnguyn/K4-Track4-NguyenPhucHuy-2A202602911-Day2-Deeplearning-Advance/blob/main/submissions/2A202602911_nguyenphuchuy/code/lab_day2.ipynb
```

Notebook tự clone repo, tải DeepWeeds (Zenodo, kiểm tra MD5), rồi chạy Bước 0→5.
Mọi bước **cache-aware**: chạy lại chỉ huấn luyện phần còn thiếu (`runs/<exp_id>/seed<k>/done.json`).
Muốn chạy từng bước: đặt `LAB_STAGE` = `b0` | `b1` | `b2` | `b3` | `b4` | `b5` trước khi chạy cell.

## Phiên bản thư viện

| | |
|---|---|
| Python | 3.13.15 |
| torch | 2.11.0+cu128 |
| timm | 1.0.29 |
| torchvision | 0.26.0+cu128 |
| GPU | Tesla T4 |

## Thứ tự chạy và seed

1. `B01`–`B06`: so sánh backbone, công thức nền `T00`, **seed 0**.
2. `T01`–`T12`: ablation công thức huấn luyện (mỗi lần khác nền một yếu tố), **seed 0**.
3. `I00`–`I08`: suy luận trên **val** (không huấn luyện lại) + đo độ trễ.
4. `T00` (mốc) và `F01` (chung kết): **seed 0, 1, 2**; **test chạy đúng một lần mỗi seed**.

## Kiểm tra tự viết

```bash
cd code && python -m unittest discover -s . -p "test_*.py" -v   # 28 test
```

Bao gồm: focal `γ=0` ≡ CE, label smoothing `ε=0` ≡ CE, CutMix λ theo diện tích hộp thực,
weight decay = 0 cho norm/bias, đóng băng backbone, gộp BN sai số ≤ 1e-4, temperature scaling giữ nguyên argmax,
và một test đầu-cuối chạy `train.run()` trên dataset nhỏ để chắc mọi artifact + file dự đoán hợp lệ với `eval.py`.

## Ghi chú

- `code/eval.py` là **bản sao y nguyên** của `eval.py` trong repo gốc (SHA-256
  `7a9f86781ac219747ea502b642db8fa8ff2e8d5796fdad9a658729546c0f4cc0`), **không sửa**.
- `starter/` trong repo vẫn nguyên bản (không sửa tại chỗ) để `python -m unittest discover -s tests` xanh.
- **Hai lỗi đã tìm và sửa trong notebook** (đều là lỗi đọc dữ liệu, không ảnh hưởng tới chỉ số đã báo cáo):
  1. Bước 4: cấu hình tổ hợp đọc từ hàng `T12` của pandas là `numpy.int64` → `json.dumps(asdict(cfg))`
     ném `TypeError: Object of type int64 is not JSON serializable`. Đã đổi về kiểu Python bằng `.item()`.
  2. Bước 5: `eval.py` ghi số ảnh test mỗi lớp vào `eval_out/<tag>_per_class.csv`, **không** có trong
     `<tag>_summary.json`; cell tạo sheet `PerClass` đọc sai chỗ → `KeyError: 'support'`. Đã đọc đúng file;
     đồng thời tra tên lớp không phân biệt hoa/thường (`Chinee apple` / `Snake weed` trong `labels.csv`).
- `curves/` có ảnh đường cong cho **từng** `exp_id` (B, T, T00, F01) + 2 ảnh tổng hợp
  (`SUMMARY_tradeoff_and_confusion.png`, `SUMMARY_ablation_delta.png`).
