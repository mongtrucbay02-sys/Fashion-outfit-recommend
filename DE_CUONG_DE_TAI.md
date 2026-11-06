# Gợi ý phối đồ từ danh mục sản phẩm
Môn Học máy và Khai phá dữ liệu. Phạm vi cập nhật ngày 05/11/2026.

## Bài toán
Người dùng chọn một món có sẵn trong danh mục Polyvore. Hệ thống lấy mã và thuộc tính món đó, tìm món bổ sung, chấm điểm và trả tối đa 5 bộ đa dạng. Giai đoạn đầu chọn áo làm món gốc, tìm quần và giày.

## Dữ liệu và mô hình
Polyvore Outfits ECCV 2018 là dataset của dự án. Ảnh sản phẩm được ResNet-50 chuyển thành vector. Triplet Loss dùng để học biểu diễn ảnh. FP-Growth khai phá các loại đồ thường xuất hiện cùng nhau, đối chiếu với Apriori. Random Forest học điểm tương thích của bộ đồ, đối chiếu với Logistic Regression. Kết hợp điểm và MMR chọn bộ cuối.

## Phân công 5 người
| Người | Nhiệm vụ | Phương pháp và bàn giao |
|---|---|---|
| 1 | Sinh ứng viên bổ sung | Lọc theo vị trí còn thiếu, lấy ứng viên từ nhóm được luật hỗ trợ và nhóm nền. Đo độ bao phủ, candidate recall và thời gian |
| 2 | Học đặc trưng ảnh sản phẩm | ResNet-50 + Triplet Loss, cosine similarity, checkpoint và vector có phiên bản |
| 3 | Khai phá luật phối đồ | Apriori/FP-Growth, support/confidence/lift, bộ luật và báo cáo đối chiếu |
| 4 | Phân loại tương thích | Logistic Regression/Random Forest, F1/ROC-AUC/confusion matrix, mô hình đã lưu |
| 5 | Xếp hạng và đánh giá kết hợp | Trọng số điểm + MMR, Hit@K/MRR, độ đa dạng, đánh giá ảnh hưởng từng thành phần |

Chuẩn bị dataset và tiền xử lý là đầu vào chung, thực hiện bằng các script riêng.

## Luồng triển khai
Chọn mã sản phẩm → lấy thuộc tính và embedding → sinh ứng viên → ghép bộ → chấm điểm tương thích → MMR → hiển thị kết quả và căn cứ giải thích.

## Kiểm tra chất lượng
Tệp disjoint gốc có giao nhau về mã món. Tiền xử lý giữ train và loại outfit bị giao nhau khỏi tập giữ lại. Báo cáo số loại bỏ trong preprocessing_report.json.
Bộ ba dương được chọn từ outfit gốc. Bộ ba âm tạo bằng thay một món cùng loại và có thể chứa nhãn nhiễu. Công bố rõ giao thức này khi báo cáo.
Dùng train để học, validation để chọn tham số và test để đánh giá cuối. Các phép so sánh dùng cùng ngân sách ứng viên.
Recall của encoder trên ảnh tăng cường cùng món đo truy xuất danh tính. Chất lượng phối đồ cần được đo riêng bằng bài toán tương thích và xếp hạng.
Giao diện được thực hiện sau khi hoàn tất huấn luyện và xem xét kết quả.

## Hiện trạng
Đã có dữ liệu Polyvore tiền xử lý, luật và baseline metadata. ResNet-50 đang huấn luyện. Tiến độ và kết quả thực tế ở reports/ và logs/.
