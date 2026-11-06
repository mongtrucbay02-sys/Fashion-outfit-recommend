# Fashion Suggest — giao diện gợi ý phối đồ

Ứng dụng local dùng chính checkpoint MLP + luật đã giữ lại sau các thí nghiệm. Chỉ có chọn món ban đầu và gợi ý; **không có phòng phối đồ thủ công, nhận diện ảnh hay thử đồ trên người**.

## Khởi chạy trên máy hiện tại

Nhấp đúp `Chay_giao_dien.cmd` tại thư mục gốc dự án. Khi xuất hiện dòng `Fashion Suggest sẵn sàng`, mở **http://127.0.0.1:8765** bằng Edge hoặc Chrome. Giữ cửa sổ máy chủ mở; nhấn Ctrl+C để dừng.

Hoặc chạy từ thư mục gốc:

```powershell
.\.venv\Scripts\python.exe -X utf8 web/server.py
```

Nếu cổng đã được dùng, dùng `--port 8766` rồi mở địa chỉ tương ứng. Không cần Flask, Node hay cài thêm thư viện web; backend dùng thư viện HTTP Python có sẵn, NumPy và PyTorch hiện có. Máy khác cần cài phụ thuộc mô hình và sao chép các dữ liệu/checkpoint tương ứng, không chỉ thư mục `web`.

## Cách dùng

1. Chọn tab Áo, Quần/váy hoặc Giày. Tìm theo tên, kiểu dáng hoặc chuyển trang.
2. Chọn một món; món này được giữ nguyên trong mọi gợi ý.
3. Có thể bật “Ưu tiên gợi ý đa dạng” để giảm lặp các món khác giữa các bộ.
4. Nhấn “Tìm bộ đồ phù hợp”. Hệ thống trả tối đa 5 bộ áo–quần/váy–giày.
5. “Xem gợi ý” mở ảnh và lý do thực sự dùng trong xếp hạng. Nhấn trái tim để lưu trên trình duyệt; nhấn lại để bỏ lưu. Không cần tài khoản.

## Dữ liệu và mô hình

- Danh mục: giao giữa `data/processed/polyvore/items.json` và 44.503 ID trong `artifacts/embedding/catalog_vectors.npz`.
- Ảnh: `data/processed/polyvore/images/`.
- Checkpoint giữ lại: `artifacts/outfit_ranker/best.pt`. Khi nạp, đối chiếu hash embedding.
- Luật: `artifacts/mining/fpgrowth_rules.json`.
- Mỗi vị trí còn thiếu lấy tối đa 48 ứng viên: ưu tiên loại có luật với món gốc và giữ ít nhất 16 lựa chọn nền ngẫu nhiên (nếu đủ danh mục). Không dùng bộ đồ đáp án từ test để sinh ứng viên giao diện.
- Chấm tối đa 2.304 bộ, kết hợp 0,75 điểm MLP và 0,25 confidence của luật phù hợp mạnh nhất. Mặc định lambda=1; bật đa dạng dùng lambda=0,9 và phạt lặp ở hai vị trí còn lại. Không diễn giải điểm này là xác suất phối đẹp.
- Lý do luật nêu đúng thuộc tính loại trang phục có mặt trong luật đã dùng. Không bịa lý do màu sắc, mùa, số đo hoặc dịp mặc.
- Các chỉ số Hit@5 trong trang “Về dự án” thuộc bài kiểm tra ứng viên kiểm soát, không đo chất lượng tìm kiếm trên toàn danh mục đang dùng ở giao diện. Bộ dữ liệu có thể chứa kết hợp không hợp mắt; cần đánh giá người dùng trước khi khẳng định chất lượng sản phẩm.

## Kiểm tra và giới hạn

```powershell
.\.venv\Scripts\python.exe -X utf8 web/test_app.py
node --check web/static/app.js
```

Đã qua ba nhóm kiểm tra tích hợp với checkpoint và ảnh thật: bộ lọc, gợi ý cho cả ba vị trí, API và chặn đường dẫn ngoài danh mục. Kiểm tra cú pháp JavaScript thành công. Bố cục có breakpoint điện thoại/tablet, trạng thái tải, lỗi, không có kết quả, bàn phím, dialog và giảm chuyển động.

Máy chủ chỉ lắng nghe `127.0.0.1`, dùng cho demo local. Không phải cấu hình triển khai public. Bộ sưu tập chỉ lưu bằng localStorage trong trình duyệt này; xóa dữ liệu trình duyệt sẽ mất bộ đã lưu.

Phong cách kem/xanh olive, typography và bố cục tham khảo nguồn người dùng cung cấp:
- https://github.com/mongtrucbay02-sys/FashionStore/blob/main/Views/Outfit/Index.cshtml
- https://github.com/mongtrucbay02-sys/FashionStore/blob/main/wwwroot/css/outfit.css

Giao diện và backend được viết riêng cho dự án Python này; không sửa hay đẩy mã lên repository FashionStore.
