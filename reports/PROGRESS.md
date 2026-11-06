# Tiến độ thực tế

Ngày cập nhật: 04/11/2026. Người dùng chọn tiếp tục trên CPU máy hiện tại.

## Đã hoàn tất

- Môi trường Python riêng; PyTorch CPU, torchvision và các thư viện đã cài.
- Polyvore: sao chép tệp hoàn chỉnh từ lần tải trước; đối chiếu kích thước/checksum với manifest nguồn chính thức.
- Fashionpedia: tải, kiểm tra CRC ZIP, giải nén ảnh và đọc kiểm tra nhãn.
- Hai tệp tiền xử lý độc lập đã chạy thành công.
- Khai phá Apriori/FP-Growth: cùng 153 tập mục phổ biến và 137 luật với cấu hình hiện tại.
- Huấn luyện Logistic Regression và Random Forest baseline chỉ dùng metadata.
- Thử nghiệm xếp hạng metadata, kết hợp điểm và MMR trên tập ứng viên kiểm soát.

## Số lượng sau tiền xử lý

| Dữ liệu | Train | Validation | Test |
|---|---:|---:|---:|
| Fashionpedia: ảnh | 45.623 | 579 | 579 |
| Fashionpedia: annotation trong phạm vi | 103.576 | 1.497 | 1.572 |
| Polyvore: món đồ hợp lệ trong phạm vi | 30.384 | 1.278 | 29.818 |
| Polyvore: bộ ba dương | 7.972 | 267 | 6.594 |
| Polyvore: bộ ba âm tổng hợp | 7.972 | 267 | 6.594 |

Fashionpedia test ở đây là nửa validation chính thức giữ lại nội bộ. Polyvore đã loại outfit bị giao nhau giữa các tập; tệp nguồn không sửa. Cấu hình loại bỏ và số lượng chi tiết ở các preprocessing_report.json.

## Kết quả baseline metadata

| Mô hình | ROC-AUC test | F1 test | Accuracy test |
|---|---:|---:|---:|
| Logistic Regression | 0,5764 | 0,6706 | 0,5225 |
| Random Forest | 0,5776 | 0,6697 | 0,5189 |

Các điểm này đo trên nhãn âm tổng hợp của dự án. Mô hình dự đoán dương nhiều; F1 tương đối cao không đồng nghĩa phân biệt tốt. ROC-AUC và confusion matrix cho thấy baseline metadata còn yếu.

Xếp hạng metadata: Hit@5 = 0,14 trên 100 truy vấn test, mỗi truy vấn có 100 bộ ứng viên và chắc chắn chứa bộ gốc. Chưa đại diện tìm kiếm trên toàn danh mục hay đánh giá thẩm mỹ.

## Huấn luyện ảnh và việc còn lại

Profile pilot đã hoàn tất mọi bước với exit code 0. Pilot giới hạn số bước/mẫu, không được coi là huấn luyện đầy đủ. Lượt ResNet-50 đầu với margin 0,3 có mean loss = 0. Mask R-CNN pilot học 4 bước, đánh giá trên 8 ảnh: box mAP test khoảng 0,0108 và mask mAP = 0; chưa đủ chất lượng sử dụng.

Đã khởi chạy profile full từ bước embedding: 3 epoch trên CPU, không giới hạn số mẫu/bước. ResNet-50 dùng margin 1,0; batch đầu có loss = 0,1983 (khác 0), khoảng 2,6 giây, 3.798 batch/epoch. Đây mới là xác nhận đã bắt đầu học, không phải kết quả cuối. Recall trên ảnh lật ngang cùng món không phải điểm phối đồ.

Pipeline chạy tuần tự: ResNet-50 → Mask R-CNN → xuất embedding → huấn luyện LR/RF có đặc trưng ảnh → đánh giá xếp hạng ảnh. Các bước metadata đã có kết quả được bỏ qua trong lần chạy này để tránh chạy lại không cần thiết. Lệnh đã khởi chạy: `scripts/run_training_pipeline.py --profile full --start-at embedding`.

Trạng thái mới nhất: `reports/pipeline_pilot.json`, `reports/pipeline_full.json` nếu đã khởi chạy, và log từng bước trong `logs/`. Checkpoint ở `artifacts/`.

Chạy `scripts/check_training_status.py` để kiểm tra các đầu ra bắt buộc. Chưa tạo giao diện. Cần hoàn tất và xem xét kết quả mô hình ảnh, mô hình tương thích dùng ảnh và xếp hạng trước khi làm giao diện.
