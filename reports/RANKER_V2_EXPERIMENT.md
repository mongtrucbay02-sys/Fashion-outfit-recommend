# Thử nghiệm cải thiện xếp hạng lần 2

Ngày 05/11/2026. Hoàn tất với exit code 0 sau **486,98 giây (8 phút 7 giây)**, chưa tính thời gian viết mã và kiểm tra. Không thay đổi mô hình cũ hay tạo giao diện.

## Kết luận

**Chưa có bằng chứng để thay MLP + luật của lần trước bằng mô hình mới.** MLP nhỏ hơn và mô hình chấm điểm từng cặp không vượt mô hình cũ trong đợt thử này. Giữ nguyên checkpoint và hệ số hiện có, không nâng số liệu 15% thành 21,6% như một cải thiện do huấn luyện.

## Phương pháp

- Giữ nguyên ResNet-50 và 44.503 embedding.
- Hai kiến trúc: MLP cả bộ (hidden=64) và ba mạng chấm cặp áo–quần, áo–giày, quần–giày (hidden=32), lấy trung bình ba điểm.
- Dropout 0,4; AdamW, learning rate 0,001, weight decay 0,01; pairwise softplus ranking loss.
- Ba tỷ lệ âm khó: 0%, 25%, 50%; âm cùng vị trí, chỉ lấy trong train, loại các bộ dương train đã biết. Âm khó là hàng xóm cosine của món bị thay, không phải nhãn không hợp do con người xác nhận.
- Sáu cấu hình × ba seed (17, 42, 83) = 18 lượt; tối đa 30 epoch/lượt, patience=6, chỉ dừng sớm sau ít nhất 10 epoch. Mỗi epoch duyệt các bộ dương train một lần theo thứ tự ngẫu nhiên.
- Dùng đủ 267 truy vấn validation; chọn checkpoint theo Hit@5, hòa dùng MRR@5. Chọn cấu hình bằng trung bình ba seed, sau đó lấy trung bình điểm sigmoid của ba checkpoint thành ensemble.
- So sánh ensemble với Logistic Regression, Random Forest và MLP cũ. Chọn mô hình và hệ số luật/MMR trên validation bằng Hit@5 + 0,05 × diversity. Lưu `selection.json` trước khi đánh giá test.
- Mỗi truy vấn có 100 ứng viên và bảo đảm bộ gốc có mặt. Sinh đúng 100 truy vấn cũ rồi thêm 500 truy vấn kế tiếp với seed 71; bộ gốc của hai nhóm không trùng. Tập train/validation/test không trùng item ID.

## Kết quả test

| Phương án | Hit@5, 100 truy vấn cũ | Hit@5, 500 truy vấn bổ sung | MRR@5, 500 | Diversity@5, 500 |
|---|---:|---:|---:|---:|
| Random Forest ban đầu | 12% | 12,2% | 0,0576 | 0,8279 |
| MLP + luật lần trước, giữ hệ số cũ | 15% | 21,6% | 0,1140 | 0,7363 |
| Logistic Regression, hệ số chọn bằng validation | 14% | 14,6% | 0,0714 | 0,8228 |
| Ensemble MLP mới, hệ số chọn bằng validation | 11% | 18,6% | 0,1003 | 0,8388 |
| Phương án được validation lần này chọn: MLP cũ thuần | 14% | 20,8% | 0,1134 | 0,7301 |

Ensemble mới dùng MLP cả bộ, hidden=64, âm ngẫu nhiên 100%; alpha=0,75 và lambda=0,9. Cấu hình này có Hit@5 validation trung bình ba seed là 21,60%, độ lệch chuẩn 0,98 điểm phần trăm. Mô hình từng cặp có trung bình thấp hơn, không được chọn.

Validation toàn bộ chọn **MLP cũ thuần**, alpha=1, lambda=1, Hit@5=22,47%. Trên 500 truy vấn bổ sung, chênh lệch Hit@5 của phương án này so với MLP + luật cũ là **−0,8 điểm phần trăm**, khoảng bootstrap 95% **[−3,4; +1,6] điểm phần trăm**. Khoảng chứa 0, chưa cho thấy khác biệt rõ. Không chọn lại cấu hình theo kết quả test và không thay cấu hình đang có chỉ để chạy theo số test.

## Giới hạn và kiểm tra

- 500 truy vấn bổ sung vẫn thuộc **cùng test split**, không phải dataset độc lập mới. Các mô hình phân loại cũ từng được đánh giá phân loại tổng hợp trên split này. Một số món đồ có thể xuất hiện trong nhiều truy vấn; bootstrap theo truy vấn có thể đánh giá thấp bất định.
- Đã dùng validation để thử nhiều checkpoint; điểm validation vẫn có thể lạc quan. Nhiều seed giúp kiểm tra dao động, không xóa hoàn toàn ảnh hưởng chọn mô hình.
- 15% và 21,6% thuộc hai nhóm truy vấn khác nhau của cùng MLP cũ; không thể diễn giải thành tăng chất lượng do lần huấn luyện này.
- Đã tái hiện đúng Hit@5, MRR@5 và diversity của MLP cũ trên 100 truy vấn cũ; Random Forest cũng tái hiện Hit@5=12%.
- Checkpoint ensemble được nạp lại từ đĩa trước đánh giá, kiểm tra hash embedding. Báo cáo đủ 18 lượt và đủ 500 kết quả hit từng truy vấn cho mỗi phương án.
- Đây là bài tìm lại bộ gốc, không phải đánh giá tỷ lệ phối đẹp.

## Tệp

- `scripts/experiment_ranker_v2.py`: huấn luyện, lựa chọn validation, đánh giá và bootstrap.
- `scripts/ranker_v2.py`: kiến trúc và nạp ensemble.
- `artifacts/outfit_ranker_v2/plan.json`: kế hoạch ghi trước khi chạy.
- `artifacts/outfit_ranker_v2/config_0.pt` đến `config_5.pt`: sáu nhóm checkpoint, mỗi nhóm ba seed.
- `artifacts/outfit_ranker_v2/ensemble.pt`: ensemble được chọn trong nhóm mô hình mới.
- `artifacts/outfit_ranker_v2/selection.json`: lựa chọn cuối trên validation (MLP cũ, không phải ensemble).
- `reports/metrics/ranking_v2.json`: báo cáo đầy đủ, lịch sử epoch, hệ số, kết quả từng truy vấn và hash ứng viên.
- `logs/ranker_v2.log`: log chạy.

Chạy lại sẽ ghi đè riêng kết quả v2:

```powershell
.\.venv\Scripts\python.exe -X utf8 -u scripts/experiment_ranker_v2.py
```

Nếu tiếp tục nghiên cứu, cần một thí nghiệm mới về đặc trưng ảnh hoặc cách học tương thích; không tiếp tục chỉnh tham số theo các kết quả test này rồi gọi đó là đánh giá độc lập.
