# Kết quả thử MLP xếp hạng bộ đồ

Ngày 05/11/2026. Giữ nguyên ResNet-50 và 44.503 vector 128 chiều; không huấn luyện lại ảnh. Mô hình cũ được giữ nguyên.

## Cách học

- Đầu vào 768 chiều: vector áo, quần, giày và tích từng cặp vector.
- MLP: 768 → 128/256 → 64 → 1; ReLU, dropout, AdamW.
- Loss: `softplus(score_negative - score_positive)`; điểm sigmoid dùng để xếp hạng, không phải xác suất phối đẹp đã hiệu chỉnh.
- 7.972 bộ dương train. Lấy mẫu lại với hoàn lại mỗi epoch; thay một vị trí bằng món cùng loại. Loại bộ âm trùng bộ dương đã biết trong train.
- Ba cấu hình, mỗi cấu hình 20 epoch, seed 42. Cấu hình 1 chỉ âm ngẫu nhiên; cấu hình 2–3 trộn 50% âm ngẫu nhiên và 50% âm lấy từ tối đa 20 hàng xóm cosine của món cùng vị trí trong train.
- Chọn checkpoint bằng Hit@5 validation, hòa thì dùng MRR@5. Chọn hệ số luật/MMR bằng Hit@5 + 0,05 × diversity trên validation. Chỉ đánh giá test sau khi khóa lựa chọn.

## Kết quả

Giữ cùng quy trình sinh 100 truy vấn, 100 bộ ứng viên/truy vấn và seed 71 như bản cũ. Bộ gốc luôn có mặt trong ứng viên.

| Mô hình | Test Hit@5 | Test MRR@5 | Test diversity@5 |
|---|---:|---:|---:|
| Random Forest + embedding cũ | 0,12 | 0,04150 | 0,8310 |
| MLP thuần | 0,14 | 0,07917 | 0,7300 |
| MLP + luật, chọn trên validation | 0,15 | 0,06917 | 0,7305 |

Checkpoint được chọn: hidden=128, dropout=0,2, tỷ lệ âm khó=0,5, epoch=18. Validation Hit@5=0,28. Hệ số alpha=0,75; lambda=1,0 nên bộ chọn chưa áp dụng phạt trùng của MMR. Thời gian thực thi script huấn luyện gồm chuẩn bị và đánh giá: 153,58 giây CPU; không bao gồm thời gian viết mã và kiểm tra độc lập.

Hit@5 tăng 3 điểm phần trăm (thêm 3/100 truy vấn trúng), nhưng độ đa dạng giảm. Đây chưa phải bằng chứng cải thiện ổn định: tập đánh giá nhỏ, một seed, validation có chọn nhiều checkpoint, và mẫu âm tổng hợp có thể vẫn là bộ phối hợp lý. Không diễn giải Hit@5 thành tỷ lệ phối đẹp. Không điều chỉnh tiếp mô hình dựa trên test trong lần thử này.

## Tệp và chạy lại

- `scripts/train_outfit_ranker.py`: huấn luyện và đánh giá; chạy lại sẽ ghi đè kết quả MLP cùng đường dẫn.
- `artifacts/outfit_ranker/best.pt`: trọng số và cấu hình, hash embedding, hệ số xếp hạng.
- `reports/metrics/ranking_mlp.json`: toàn bộ lịch sử validation và kết quả test.
- `scripts/outfit_ranker.py`: nạp checkpoint để suy luận, kiểm tra hash embedding.
- `reports/metrics/ranking_mlp_verified.json`: kiểm tra bằng chương trình đánh giá cũ với checkpoint đã lưu.
- `logs/mlp_ranking.log`, `logs/mlp_verification.log`: nhật ký.

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/train_outfit_ranker.py
.\.venv\Scripts\python.exe -X utf8 scripts/evaluate_recommendation.py --embeddings artifacts/embedding/catalog_vectors.npz --mlp-checkpoint artifacts/outfit_ranker/best.pt
```

Chưa thay đổi pipeline full để tự chọn MLP, chưa triển khai giao diện, chưa cập nhật gói Colab/bàn giao. Đây là thí nghiệm cải thiện riêng, có thể sử dụng checkpoint qua bộ nạp mới.
