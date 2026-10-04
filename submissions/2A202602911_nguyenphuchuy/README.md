# Submission — Lab Day 2 (Track 4) · Nguyễn Phúc Huy · 2A202602911

**Trạng thái:** bộ code và notebook đã hoàn thiện và đã kiểm thử; các bước cần GPU (Bước 1–4)
đang chờ chạy đủ trên T4. Notebook tự sinh `results.xlsx`, `curves/`, `predictions/`, `report.md`
và ghi đè chính `README.md` này bằng bản có đầy đủ số liệu chạy thật.

## Chạy lại (một lần, trên GPU)

1. Mở notebook trên Google Colab (repo công khai nên link mở thẳng):

   `https://colab.research.google.com/github/Yuhnguyn/K4-Track4-NguyenPhucHuy-2A202602911-Day2-Deeplearning-Advance/blob/main/submissions/2A202602911_nguyenphuchuy/code/lab_day2.ipynb`

2. *Runtime → Change runtime type → T4 GPU*.
3. *Run all*. Notebook sẽ: clone repo → tải DeepWeeds (Zenodo, kiểm tra MD5) → chạy Bước 0→5.
   Mọi bước **cache-aware** (`runs/<exp_id>/seed<k>/done.json`): chạy lại chỉ huấn luyện phần còn thiếu.
   Chạy từng bước bằng biến môi trường `LAB_STAGE` = `b0|b1|b2|b3|b4|b5` (rỗng/tất cả = chạy hết),
   và `LAB_ONLY`/`LAB_SEEDS` để giới hạn từng `exp_id`/seed.
4. Ô cuối gói toàn bộ thư mục nộp thành `submission_2A202602911_nguyenphuchuy.tar.gz` để tải về.

## Nội dung thư mục

| File | Nội dung |
|---|---|
| `code/dataset.py` | đọc/chia dữ liệu (S1–S6), 4 kiểm tra bắt buộc, transform + 6 mức augmentation, `Dataset`, `DataLoader` (kể cả balanced sampler) |
| `code/model.py` | `timm` backbone, đóng băng, 4 nhóm tham số (wd = 0 cho norm/bias), đếm params/GMAC (fvcore) |
| `code/losses.py` | CE · label smoothing · focal · CE có trọng số lớp (beta=0 và class-balanced), Mixup/CutMix |
| `code/train.py` | một hàm `run(cfg)` cho mọi thí nghiệm: AMP, warmup+cosine, EMA, chọn checkpoint theo macro-F1 val, ghi config/history/checkpoint/logit/prediction |
| `code/inference.py` | TTA lật & multi-crop & multi-scale, gộp xác suất/logit, ensemble, temperature scaling, gộp BatchNorm |
| `code/benchmark.py` | đo độ trễ đúng cách: warmup, `cuda.synchronize`, ≥ 50 lượt, p50/p95/p99, độ trễ TTA K view |
| `code/test_code.py` | **28 test tự viết** (xem dưới) |
| `code/eval.py` | bản sao **y nguyên** của `eval.py` gốc, không sửa (SHA-256 `7a9f8678…4cc0`) |
| `code/lab_day2.ipynb` | notebook chạy toàn bộ Bước 0→5 |

## Kiểm thử

```bash
cd code && python -m unittest discover -s . -p "test_*.py" -v     # 28 test
python -m unittest discover -s ../../tests                        # test của repo (38 test) vẫn xanh
```

`test_code.py` kiểm tra đúng những chỗ dễ sai: focal `γ=0` ≡ CE (sai số < 1e-6), label smoothing
`ε=0` ≡ CE, **CutMix: λ khớp diện tích hộp thực sau khi cắt ra ngoài biên**, `mixed_loss` là tổng
có trọng số, weight decay = 0 cho norm/bias, head có LR gấp 10 backbone, đóng băng backbone +
giữ BN ở eval, warmup→cosine về ~0, EMA đúng công thức, gộp BatchNorm sai số ≤ 1e-4,
temperature scaling giảm NLL và **không đổi argmax**, và một test **đầu-cuối** chạy `train.run()`
trên dataset nhỏ để chắc mọi artifact + file dự đoán qua được `eval.read_pred`/`check_against_csv`.

## Ghi chú

- `starter/` trong repo vẫn **nguyên bản** (không sửa tại chỗ) nên `python -m unittest discover -s tests`
  vẫn xanh; toàn bộ phần hoàn thiện nằm trong `code/`.
- `eval.py` không bị sửa (so SHA-256 với bản gốc trong repo).
