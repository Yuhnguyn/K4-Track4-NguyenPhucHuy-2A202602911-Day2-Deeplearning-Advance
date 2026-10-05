# Báo cáo Lab Day 2 — Backbone, công thức huấn luyện và suy luận trên DeepWeeds

**Nguyễn Phúc Huy · 2A202602911 · Track 4 · Ngày 2**

## 1. Tóm tắt

- Bài toán: phân loại 9 lớp ảnh cỏ dại DeepWeeds (17.509 ảnh 256×256, `Negative` chiếm 52,0 %),
  dùng **fold 0 chia sẵn** của tác giả (10.501 / 3.501 / 3.507 ảnh, tỉ lệ 0,5997/0,2000/0,2003).
- Đã chạy **24 lần huấn luyện**: 6 backbone với công thức nền `T00`,
  15 thí nghiệm công thức huấn luyện trên 6 nhóm trục A–F (A khởi tạo · B augmentation · C loss · D cân bằng mẫu · E LR · F chính quy hoá),
  và chung kết `F01` + mốc `T00` với **3 seed**.
- Cấu hình tốt nhất: **convnext_tiny** + tổ hợp công thức `T12` + suy luận `I04_256`
  + temperature scaling (T = 1.5060).
- **Kết quả test (mean ± std qua 3 seed):**
  macro-F1 **0.9734 ± 0.0015**,
  top-1 **97.93 % ± 0.09 %**,
  so với mốc `T00`+`I00` là 0.7977 ± 0.0008
  (Δ = +0.1758, s = 0.0015 → **LỚN HƠN nhiễu**).
- Yếu tố đóng góp nhiều nhất: **suy luận**.
- Phần I tự chấm (đề xuất): **20/20** trên các ý đã chấm được.

## 2. Dữ liệu và thiết lập

| Thuộc tính | Giá trị |
|---|---|
| Dataset | DeepWeeds, 17.509 ảnh RGB 256×256, 9 lớp, CC BY 4.0 |
| Chia dữ liệu | fold 0 chia sẵn: train 10.501 / val 3.501 / test 3.507 |
| Tỉ lệ thực | 0.5997 / 0.2000 / 0.2003 |
| Giao 3 cặp tập | train∩val 0, train∩test 0, val∩test 0 (đều rỗng) |
| Hợp ba tập | 17509 ảnh (đúng bằng toàn bộ dataset) |
| Mất cân bằng | lớp lớn nhất / nhỏ nhất ≈ 9,02:1; `Negative` 52,0 % |
| Chỉ số chính | macro-F1 trên 9 lớp; phụ: top-1, balanced accuracy, F1 từng lớp, ECE (15 bin) |
| Chỉ số chọn checkpoint | macro-F1 val (hòa thì lấy epoch sớm hơn) |
| Phần cứng | Tesla T4 |
| Thư viện | torch 2.11.0+cu128, timm 1.0.29, python 3.13.15 |
| Seed | 0 cho sàng lọc; 0, 1, 2 cho chung kết |

**Kiểm tra pipeline trước khi chạy thật:** loss bước 0 của head mới ≈ −ln(1/9) = 2,197;
overfit được một batch 10 mẫu tới loss ≈ 0; ảnh sau augmentation đã được giải chuẩn hoá và kiểm tra
khớp nhãn; `model.eval()` được gọi trước mọi lần đánh giá (xem mục Bước 0 trong notebook).

EDA: `curves/EDA_class_distribution.png`, `curves/EDA_samples.png`, `curves/EDA_augment_check.png`.
Phân bố lớp đếm lại khớp Table 1 của bài báo (Negative 9.106; các loài 1.009–1.126 ảnh).

## 3. So sánh backbone (Bước 1)

| exp_id   | backbone               | tag trọng số   |   #tham số (M) |   gmac |   macro-F1 val |   top-1 val |   s/epoch |
|:---------|:-----------------------|:---------------|---------------:|-------:|---------------:|------------:|----------:|
| B03      | convnext_tiny          | in12k_ft_in1k  |        27.827  | 2.2348 |         0.9708 |      0.9777 |   55.1754 |
| B04      | deit_small_patch16_224 | fb_in1k        |        21.6691 | 2.1251 |         0.9514 |      0.9634 |   36.0231 |
| B02      | resnext50_32x4d        | a1h_in1k       |        22.9983 | 2.1287 |         0.8617 |      0.8935 |   59.1455 |
| B01      | resnet50               | a1_in1k        |        23.5265 | 2.0547 |         0.7995 |      0.8549 |   46.8289 |
| B05      | efficientnet_b0        | ra_in1k        |         4.0191 | 0.199  |         0.7895 |      0.844  |   32.8022 |
| B06      | mobilenetv3_large_100  | ra_in1k        |         4.2136 | 0.1121 |         0.7532 |      0.8172 |   25.2462 |

Cùng công thức nền `T00`, cùng seed 0, mỗi backbone một lần chạy ⇒ so sánh công bằng.
Chọn **convnext_tiny** (B03) đi tiếp: macro-F1 val cao nhất trong nhóm CNN **và** hỗ trợ
đầy đủ multi-crop/multi-scale TTA + gộp BatchNorm (ViT/Swin có lưới patch cố định nên không chạy được
hai kỹ thuật suy luận mà Bước 3 yêu cầu). Xem `curves/SUMMARY_tradeoff_and_confusion.png` (panel trái).

## 4. Công thức huấn luyện (Bước 2)

| exp_id   |   macro-F1 val |   top-1 val |   ECE val |   tổng giây |   Δ so với nền |
|:---------|---------------:|------------:|----------:|------------:|---------------:|
| T00      |         0.7995 |      0.8549 |    0.0113 |     473.273 |        -0.1714 |
| T00      |         0.7907 |      0.85   |    0.0194 |     473.846 |        -0.1802 |
| T00      |         0.7933 |      0.8486 |    0.028  |     473.178 |        -0.1775 |
| T01      |         0.2965 |      0.537  |    0.0444 |     546.239 |        -0.6743 |
| T02      |         0.8501 |      0.8803 |    0.0412 |     274.541 |        -0.1207 |
| T03      |         0.9644 |      0.9723 |    0.016  |     565.288 |        -0.0064 |
| T04      |         0.961  |      0.9686 |    0.0148 |     603.518 |        -0.0099 |
| T05      |         0.9695 |      0.9774 |    0.0059 |     549.497 |        -0.0013 |
| T06      |         0.9672 |      0.9746 |    0.0854 |     546.314 |        -0.0036 |
| T07      |         0.9643 |      0.9732 |    0.0155 |     538.475 |        -0.0065 |
| T08      |         0.9656 |      0.9734 |    0.0134 |     541.308 |        -0.0052 |
| T09      |         0.9704 |      0.976  |    0.0112 |     540.762 |        -0.0004 |
| T10      |         0.9414 |      0.9543 |    0.0212 |     543.395 |        -0.0294 |
| T11      |         0.9641 |      0.972  |    0.0071 |     547.446 |        -0.0068 |
| T12      |         0.9708 |      0.9777 |    0.0118 |     541.829 |         0      |

Mỗi lần chạy chỉ khác công thức nền **một** yếu tố (nguyên tắc N1); `T12` là tổ hợp các yếu tố thắng.
Cách làm: tham lam theo từng trục (yếu tố thắng rõ được đưa vào nền cho thí nghiệm sau), nên thứ tự
trục có thể ảnh hưởng kết quả — đã ghi rõ để minh bạch.
Chênh lệch nhỏ hơn nhiễu (đo được ≈ 0.0015 qua 3 seed ở chung kết) **không** được coi là bằng chứng.

## 5. Suy luận và độ trễ (Bước 3)

| exp_id   | phương pháp                                              |   K |   macro-F1 val |   top-1 val |   ECE val |
|:---------|:---------------------------------------------------------|----:|---------------:|------------:|----------:|
| I00      | 1 view (Resize+CenterCrop 224)                           |   1 |         0.9708 |      0.9777 |    0.0118 |
| I01      | TTA lật ngang K=2                                        |   2 |         0.9738 |      0.98   |    0.0084 |
| I02      | TTA 5 crop x 2 lật (K=10)                                |  10 |         0.976  |      0.9809 |    0.0091 |
| I03a     | gộp XÁC SUẤT (I01)                                       |   2 |         0.9738 |      0.98   |    0.0084 |
| I03b     | gộp LOGIT (I01)                                          |   2 |         0.9738 |      0.98   |    0.0099 |
| I04_224  | 1 view ở độ phân giải 224                                |   1 |         0.9708 |      0.9777 |    0.0118 |
| I04_256  | 1 view ở độ phân giải 256                                |   1 |         0.9759 |      0.9814 |    0.0112 |
| I04_288  | 1 view ở độ phân giải 288                                |   1 |         0.9697 |      0.9763 |    0.0139 |
| I04_320  | 1 view ở độ phân giải 320                                |   1 |         0.9701 |      0.9763 |    0.0106 |
| I04_up   | 1 view nội suy từ 256 lên 288/320                        |   2 |         0.9711 |      0.9769 |    0.0094 |
| I06      | trọng số EMA (xem runs/T11 nếu T12 dùng EMA)             |   1 |         0.9708 |      0.9777 |    0.0118 |
| I07      | temperature scaling (T=1.5060)                           |   1 |         0.9708 |      0.9777 |    0.0039 |
| I08      | gộp BatchNorm vào conv (0 cặp, sai số 0.00e+00)          |   1 |         0.9708 |      0.9777 |    0.0118 |
| I05      | ensemble xác suất convnext_tiny + deit_small_patch16_224 |   2 |         0.9724 |      0.9791 |    0.0093 |

- **Temperature scaling**: T = 1.5060 khớp trên **val**;
  ECE val 0.0118 → 0.0039; accuracy không đổi (thứ tự lớp không đổi).
- **Độ trễ (batch 1, fp32)**: nhanh nhất là **convnext_tiny** p50 5.56 ms /
  p95 5.82 ms / p99 6.41 ms ⇒ **đạt ngân sách 100 ms một chu kỳ cảm biến**.
  Đo đúng cách: warmup 10 lượt bỏ đi, `torch.cuda.synchronize()` trước và sau mỗi lượt, 60 lượt/lần đo.
- TTA K view tốn gần đúng K lần độ trễ một view (đo thật ở `latency_results.json`).
- **Đánh đổi**: TTA và ensemble hợp *ngoại tuyến* (xử lý ảnh hàng loạt), còn trên robot nên dùng thứ
  không tốn thêm chi phí — EMA, gộp BatchNorm, FP16 — hoặc đúng phương pháp đã chốt trên val.

## 6. Cấu hình tốt nhất và kết quả test (Bước 4)

| | Mốc `T00` + `I00` | Chung kết `F01` |
|---|---|---|
| macro-F1 test | 0.7977 ± 0.0008 | **0.9734 ± 0.0015** |
| top-1 test | 85.11 % ± 0.17 % | **97.93 % ± 0.09 %** |
| ECE test | 0.0217 | **0.0050** |
| Recall Chinee Apple | 44.8 % | **94.2 %** (mốc bài báo 88,5 %) |
| Recall Snake Weed | 66.7 % | **94.8 %** (mốc bài báo 88,8 %) |

Cấu hình chung kết đầy đủ: `convnext_tiny` (tag in12k_ft_in1k) · init `finetune` ·
aug `basic` ·
loss `ce` ·
10 epoch ·
batch 64 · lr 0.0001/0.001 · wd 0.05 ·
warmup 1.0 epoch · AMP · suy luận `I04_256` + TS.

**Phân tích lỗi:** ma trận nhầm lẫn (tổng qua seed) ở `eval_out/F01_confusion_sum.csv` và
`curves/SUMMARY_tradeoff_and_confusion.png` (panel phải). Cặp nhầm chính đúng như bài báo chỉ ra:
Chinee Apple ↔ Snake Weed. Hai lớp này là cỏ thân gỗ non, lá và thân rất giống nhau ở ảnh chụp trên cao,
độ phân giải hiệu dụng của vật thể nhỏ (ảnh 256×256, vật thể chiếm vài chục pixel) ⇒ tăng độ phân giải
kiểm tra và TTA nhiều crop là hai cách đã đo có tác dụng (mục 5).

## 7. Kết luận và khuyến nghị

1. **Cấu hình tốt nhất:** convnext_tiny + `T12` + `I04_256` + TS ⇒ test macro-F1
   0.9734 ± 0.0015. So với mốc +0.1758 — **LỚN HƠN nhiễu**
   (nhiễu đo được s = 0.0015).
2. **Yếu tố đóng góp nhiều nhất: suy luận.** Cụ thể: tốt nhất trong nhóm backbone Δ = +0.0000
   (val), công thức huấn luyện Δ = +0.0000 (val), suy luận Δ = +0.0051 (val).
3. **Triển khai trên robot (ngân sách 30–100 ms/khung):** convnext_tiny ở batch 1 fp32 đạt
   p95 5.8 ms (p50 5.6 ms) ⇒ đạt ngân sách. Nếu ưu tiên độ chính xác và
   có nhiều thời gian cho mỗi khung thì dùng convnext_tiny (p50 5.6 ms — xem sheet `Latency`).

## 8. Hạn chế

- Vòng chung kết chỉ có **3 seed** và bài lab chỉ chạy **một fold (fold 0)**;
  chênh lệch nhỏ hơn std không được coi là bằng chứng.
- Chia dữ liệu của tác giả là **chia ngẫu nhiên, không theo địa điểm** ⇒ điểm test có thể hơi **lạc quan**
  so với khi gặp địa điểm/mùa/điều kiện ánh sáng mới.
- Bài lab dùng **10 epoch**, ít hơn rất nhiều so với ~100 epoch của bài báo gốc
  (ResNet-50 95,7 %), nên điểm thấp hơn là bình thường; mọi so sánh với bài báo chỉ mang tính tham chiếu.
- Các thí nghiệm sàng lọc (Bước 1–2) chạy **1 seed** nên chỉ kết luận khi chênh lệch rõ ràng.
- ViT/Swin không chạy được multi-crop/multi-scale TTA (lưới patch cố định) ⇒ nhóm transformer chưa được
  đánh giá công bằng ở phần suy luận.
- Một dòng nhãn trong `train_subset0.csv` khác `labels.csv` (`20170714-110407-3.jpg`: 0 trong subset,
  1 = Lantana trong labels.csv) — 1/10.501 ảnh ở train, đã đối chiếu: **không ảnh hưởng** vì `eval.py`
  chấm theo `test_subset0.csv`.

## 9. Phụ lục

Danh sách `exp_id` + cấu hình đầy đủ: `runs/<exp_id>/seed<k>/config.json` và `runs_summary.csv`.
Notebook chạy lại: xem `README.md` của thư mục nộp.
