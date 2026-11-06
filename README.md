
# Gợi ý phối đồ thời trang — Học máy và Khai phá dữ liệu

Phương án nghiên cứu: [DE_CUONG_DE_TAI.md](DE_CUONG_DE_TAI.md).

## Lưu ý khi clone từ GitHub

Repository chứa mã nguồn, notebook, giao diện và báo cáo thí nghiệm. Dataset, ảnh, môi trường `.venv`, checkpoint và vector không được đưa lên Git. Để chạy trên máy khác, cần chuẩn bị dữ liệu và các artifact theo [hướng dẫn giao diện](web/README.md) hoặc chạy quy trình huấn luyện; clone mã nguồn chưa đủ để chạy gợi ý ngay. Các báo cáo là kết quả lần chạy đã lưu, không phải trạng thái huấn luyện trực tiếp trên máy mới.

**Quy ước chạy dự án:** toàn bộ tải dataset, tiền xử lý, khai phá luật, huấn luyện và đánh giá chạy trên Google Colab. Máy Windows chỉ dùng để chuẩn bị/mở notebook và xem kết quả đã lưu. Hướng dẫn: [COLAB_GUIDE.md](COLAB_GUIDE.md), notebook: [notebooks/colab_training.ipynb](notebooks/colab_training.ipynb).

## Trình tự thực hiện

1. Tải/kiểm tra dữ liệu gốc.
2. Chạy hai tệp tiền xử lý riêng.
3. Huấn luyện, đánh giá và xem xét kết quả.
4. Chỉ tạo giao diện sau khi bước mô hình đã hoàn tất và được kiểm tra.

Đã có giao diện gợi ý local trong `web/`, dùng checkpoint MLP và luật thật sau khi hoàn tất huấn luyện/đánh giá. Nhấp đúp `Chay_giao_dien.cmd`, rồi mở `http://127.0.0.1:8765`. Xem [hướng dẫn giao diện](web/README.md). Không có phòng phối đồ thủ công hay thử đồ trên ảnh. Kết quả các thí nghiệm xem `reports/MLP_IMPROVEMENT.md` và `reports/RANKER_V2_EXPERIMENT.md`.

## Tệp độc lập

| Tệp | Chức năng |
|---|---|
| scripts/download_datasets.py | Tải tiếp dữ liệu, kiểm tra ZIP và manifest/checksum |
| scripts/preprocess_polyvore.py | Tách ảnh, tạo bảng món đồ/bộ ba và kiểm tra rò rỉ mã món giữa các tập |
| scripts/train_rules.py | Apriori và FP-Growth với cùng thiết lập |
| scripts/train_compatibility.py | Logistic Regression và Random Forest; metadata baseline hoặc thêm embedding ảnh |
| scripts/train_embedding.py | Fine-tune ResNet-50 bằng Triplet Loss |
| scripts/export_embeddings.py | Xuất vector gắn với checkpoint đã huấn luyện |
| scripts/evaluate_recommendation.py | Kết hợp điểm, MMR và đánh giá Top-5 |
| scripts/run_training_pipeline.py | Chạy lần lượt, ghi log và trạng thái từng bước |

## Máy Windows

Môi trường `.venv` cũ chỉ phục vụ kiểm tra/phát triển và không thuộc quy trình huấn luyện đã chốt. `requirements.lock.txt` là khóa phụ thuộc Windows; Colab cài các thư viện trong notebook và dùng PyTorch/torchvision CUDA có sẵn của runtime. Không chạy các lệnh huấn luyện PowerShell phía dưới cho quy trình chính.

## Khởi chạy

Mở `notebooks/colab_training.ipynb` bằng Google Colab, chọn GPU, rồi chạy các cell theo thứ tự. Notebook chạy tải dataset, hai bước tiền xử lý, các thí nghiệm baseline và toàn bộ pipeline mô hình ảnh trên Colab. Profile chính `full` không giới hạn mẫu/bước và dùng 3 epoch mặc định; thời lượng phụ thuộc GPU/runtime. Xem cách khởi chạy lại sau khi phiên Colab ngắt trong [COLAB_GUIDE.md](COLAB_GUIDE.md).

## Dữ liệu và giới hạn kết quả

- Polyvore lấy từ `mvasil/polyvore-outfits`, giữ metadata và tệp chính thức trong `data/raw/polyvore`. Dùng đăng nhập Hugging Face đã lưu; không ghi token vào mã nguồn hoặc chat.
- Kiểm tra tệp disjoint tải về phát hiện ID giao nhau: train/valid 3.781, train/test 84, valid/test 34. Tiền xử lý giữ train và loại toàn bộ outfit ở tập giữ lại có món thuộc các tập gốc ưu tiên trước. Kết quả là bộ chia riêng đã kiểm tra; tệp gốc không bị sửa. Báo cáo nêu số loại bỏ.
- Bài toán cơ bản chọn một bộ ba áo–quần–giày có thứ tự xác định từ mỗi outfit đủ điều kiện. Nhãn dương là phép chiếu bộ gốc; nhãn âm là thay món cùng loại tổng hợp, có thể nhiễu. Không gọi điểm đánh giá này là điểm benchmark chính thức.
- Metadata baseline dùng loại chi tiết và cặp loại, chưa có đặc trưng ảnh; được lưu riêng khỏi mô hình dùng embedding.
- Triplet positive dùng ảnh tăng cường cùng món. Recall@K dùng ảnh lật ngang của cùng danh tính, chưa là thước đo thẩm mỹ hay chất lượng chuyển miền từ ảnh người mặc.
- MMR đo độ lặp ở quần và giày, không tính chiếc áo gốc chung. Đánh giá xếp hạng dùng tập ứng viên kiểm soát đã chứa đáp án gốc; không đo khả năng tìm đáp án từ toàn bộ danh mục.
- Train chỉ dùng train, lựa chọn tham số trên validation. Test dùng cho báo cáo sau khi chốt. Điểm tương thích của một bộ không phải độ chính xác của mô hình.

Dữ liệu và checkpoint không đưa vào Git. Manifest, báo cáo tiền xử lý, cấu hình, phiên bản và kết quả thực nghiệm là căn cứ kiểm tra tiến độ.
