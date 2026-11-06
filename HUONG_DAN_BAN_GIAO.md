# Bàn giao huấn luyện phối đồ từ danh mục
Dùng dataset Polyvore Outfits và các script trong gói mã.

## Chạy trên máy hoặc Colab
Chọn GPU nếu có. Cài thư viện và đăng nhập tài khoản Hugging Face đã được cấp quyền truy cập mvasil/polyvore-outfits.

    python -m pip install -r requirements.txt
    hf auth login
    python scripts/download_datasets.py --dataset polyvore
    python scripts/preprocess_polyvore.py
    python -u scripts/run_training_pipeline.py --profile full

## Tiếp tục ResNet-50
Chép last.pt, best.pt và training_report.json vào artifacts/embedding/full trước khi chạy:

    python -u scripts/run_training_pipeline.py --profile full --start-at embedding --resume artifacts/embedding/full/last.pt

Checkpoint giữa epoch giữ trọng số và optimizer, rồi bắt đầu lại epoch đang dở. Các bước tiếp theo gồm xuất vector, huấn luyện tương thích và đánh giá xếp hạng. Chuẩn bị các kết quả luật/metadata khi bắt đầu trực tiếp từ embedding.
Người nhận gửi lại checkpoint, vector, mô hình tương thích và reports/ sau khi hoàn tất.
Trên Colab, lưu checkpoint vào Drive. Các phiên trên nhiều máy chạy độc lập, cần thống nhất người phụ trách trước khi ghép kết quả.

## Notebook
Huan_luyen_Colab.ipynb chứa mã để huấn luyện ResNet-50 độc lập.
notebooks/colab_training.ipynb chạy toàn bộ các bước từ thư mục dự án trên Drive.
