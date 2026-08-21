# Adaptive Canvas OCR CCCD - mục tiêu 5-6 giây

## Cài đặt

Giải nén ZIP và chép đè toàn bộ vào:

`C:\Users\Ahn Tuaann\cccd-ai-service`

Khởi động lại dịch vụ. Gói này là bản cộng dồn và thay thế các ZIP trước.

## Cách tối ưu

EasyOCR trước đây dùng canvas 2304 và tỷ lệ phóng 1.35 cho mọi ảnh thẻ. Với
ảnh thẻ đã crop lớn 1775 x 1119, việc phóng này làm detector xử lý nhiều pixel
hơn nhưng không tạo thêm chi tiết thật.

Cấu hình mới chọn tự động:

- Thẻ lớn từ 1500 px, cạnh ngắn từ 850 px: canvas 1920, mag_ratio 1.08.
- Thẻ trung bình từ 1150 px, cạnh ngắn từ 650 px: canvas 2112,
  mag_ratio 1.20.
- Thẻ nhỏ hoặc không đọc được kích thước: canvas 2304, mag_ratio 1.35.

Kích thước được đọc từ header ảnh, không giải mã ảnh thêm một lần.

## Kết quả kỳ vọng

Với ảnh mẫu 1775 x 1119:

- Diện tích canvas detector giảm khoảng 30,6%.
- `initialFullCardOcr` dự kiến giảm từ khoảng 6,8 giây xuống 4,5-5,3 giây.
- Kết hợp Spatial Fast Path, tổng thời gian dự kiến khoảng 5-6,5 giây trên
  cùng máy sau khi model đã warm-up.

Đây là ước tính theo lượng pixel xử lý. Hãy đo từ request thứ hai trở đi và
kiểm tra `metadata.processingStagesMs.initialFullCardOcr` để lấy kết quả thực
tế trên máy chạy dịch vụ.
