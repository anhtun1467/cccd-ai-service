# Fast Path OCR CCCD dưới 10 giây - 19/08/2026

## Cài đặt

Giải nén toàn bộ ZIP và chép đè vào thư mục:

`C:\Users\Ahn Tuaann\cccd-ai-service`

Sau đó khởi động lại dịch vụ. Đây là gói cộng dồn, thay thế các gói trước.

## Tối ưu mới

Log thực tế cho thấy OCR toàn thẻ mất khoảng 6,8 giây, còn một trường quê
quán bị OCR vùng 5 lần làm tăng thêm khoảng 3,6 giây. Fast Path mới sử dụng
ngay địa chỉ do Spatial OCR khôi phục khi thỏa mãn đồng thời:

- Nguồn kết quả là `SPATIAL_OCR`.
- Địa chỉ được tách theo nhãn và tọa độ text box.
- Giá trị vượt qua validator địa chỉ nghiêm ngặt.
- Không chứa nhãn OCR, ngày tháng hoặc chuỗi số dài bị dính vào địa chỉ.

Nếu một trong các điều kiện không đạt, pipeline vẫn chạy OCR vùng để bảo toàn
độ chính xác.

## Kết quả mong đợi với mẫu đã cung cấp

- `fieldOcrAttemptCount`: từ `5` xuống `0`.
- `fieldCropAndOcr`: từ khoảng `3629 ms` xuống còn chủ yếu thời gian cắt ảnh
  và ghi ảnh debug.
- `processingTime`: ước tính từ `11,091 giây` xuống khoảng `7,4-7,8 giây`
  trên cùng máy và cùng ảnh.
- `metadata.fieldOcrSkippedByValidatedSpatialOcr` chứa
  `placeOfOrigin` khi Fast Path được kích hoạt.

Thời gian thực tế còn phụ thuộc CPU, trạng thái model đã warm-up và tốc độ ổ
đĩa. Nên đo từ request thứ hai trở đi vì request đầu có thể gồm thời gian nạp
model EasyOCR.
