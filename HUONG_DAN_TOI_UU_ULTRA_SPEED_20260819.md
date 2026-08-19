# Tối ưu tốc độ OCR CCCD - 19/08/2026

## Cài đặt

1. Dừng dịch vụ đang chạy.
2. Giải nén toàn bộ ZIP vào thư mục gốc dự án:
   `C:\Users\Ahn Tuaann\cccd-ai-service`
3. Chọn ghi đè các file trùng tên.
4. Cài lại dependency nếu môi trường chưa đủ:
   `pip install -r requirements.txt`
5. Khởi động lại dịch vụ.

Gói này là bản cộng dồn và thay thế các gói sửa trước đó.

## Các thay đổi tốc độ

- Không OCR lại vùng số CCCD, ngày sinh, giới tính, quốc tịch và ngày hết
  hạn nếu OCR toàn thẻ đã trả về giá trị qua validator.
- Vẫn OCR riêng họ tên, quê quán và nơi thường trú khi chưa có nguồn QR xác
  thực để bảo toàn dấu tiếng Việt và nội dung nhiều dòng.
- Khi QR hợp lệ, hợp nhất tập trường bỏ qua của QR với tập trường đã được OCR
  toàn thẻ xác thực. Trường hợp đầy đủ thường chỉ còn OCR vùng quê quán.
- Giới hạn sửa xiên dư còn tối đa một ứng viên OCR toàn thẻ.
- Bỏ hẳn lượt sửa xiên khi OCR đã mạnh và góc xiên rất nhỏ.
- Giảm canvas EasyOCR toàn thẻ từ 2560/1.50 xuống 2304/1.35 vì ảnh đầu vào
  đã được chuẩn hóa gần 1600 px.

## Theo dõi hiệu năng

Trong response, kiểm tra:

- `metadata.processingStagesMs.fieldCropAndOcr`
- `metadata.fullCardOcrAttemptCount`
- `metadata.fieldOcrAttemptCount`
- `metadata.fieldOcrSkippedByValidatedFullCard`
- `metadata.qrFastPath.skippedFieldOcr`

Nếu ảnh rõ và dữ liệu toàn thẻ hợp lệ, số lượt field OCR phải giảm rõ rệt.
Tên và địa chỉ vẫn có thể phát sinh nhiều hơn một lượt nếu kết quả đầu chưa
đạt validator hoặc chưa khớp nguồn tham chiếu.

## Phạm vi an toàn

Tối ưu chỉ bỏ OCR vùng đối với trường có khuôn dạng chặt và đã qua validator.
Không bỏ OCR vùng cho họ tên hoặc địa chỉ chỉ vì chuỗi có nội dung. Khi giá
trị toàn thẻ không hợp lệ hoặc bị thiếu, pipeline tự động quay lại OCR vùng
như trước.
