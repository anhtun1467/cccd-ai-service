# Bản vá lỗi đọc sai do lệch dòng OCR

## Cài đặt

Giải nén gói vá vào thư mục gốc dự án và cho phép ghi đè, sau đó khởi
động lại API:

```powershell
Expand-Archive -LiteralPath .\cccd_ocr_fix_doc_dong_20260809.zip -DestinationPath . -Force
python -m pytest
python -m uvicorn app.main:app --reload
```

## Nguyên nhân đã xử lý

- Khối chữ trên các thẻ thử nghiệm lệch dọc khoảng `-42` đến `+42` pixel
  so với tọa độ crop cố định, khiến crop tên hoặc địa chỉ đọc sang dòng kế.
- Bộ ghép dòng có thể trộn cột hạn sử dụng bên trái vào nơi thường trú bên
  phải khi hai box có cùng cao độ.
- Validator cũ chủ yếu kiểm tra hình thức nên một địa chỉ bị ghép nhầm vẫn
  có thể được báo thành công.

## Thay đổi chính

- Tự định vị khối chữ bằng box số CCCD 12 chữ số và dịch toàn bộ crop theo
  độ lệch thực tế của từng ảnh.
- Dựng lại quê quán/nơi thường trú trực tiếp từ tọa độ box OCR, sắp theo
  trục X và tách riêng cột hạn sử dụng.
- Thêm biến thể phóng lớn, khử nhiễu nhẹ và tăng tương phản cho họ tên và
  hai trường địa chỉ.
- Loại kết quả có dấu hiệu đọc nhầm từ trường bên cạnh; không tự bịa trường
  còn thiếu.
- Kết quả thiếu hoặc đáng ngờ trả `status = OCR_PARTIAL`,
  `metadata.reviewRequired = true` và `imageQuality.decision = REVIEW_REQUIRED`
  thay vì báo `OCR_SUCCESS`.

## Kiểm chứng

- 49 kiểm thử thuần Python liên quan đến hợp nhất dữ liệu, validator, lệch
  crop và box không gian đã đạt.
- Phát lại 13 JSON hợp lệ trong bộ gửi kèm: 12 kết quả đầy đủ qua validator.
- Ca `465bbf...` trong JSON cũ không còn trả địa chỉ cư trú sai; hệ thống
  để trường này rỗng và yêu cầu kiểm tra. Lần OCR mới sẽ dùng crop đã dịch
  xuống theo độ lệch `+42` pixel để đọc đúng vùng đáy thẻ.

Môi trường dùng để đóng gói không có OpenCV/EasyOCR nên không chạy lại mô
hình OCR ảnh tại đây; phần phát lại dùng toàn bộ box và ứng viên OCR đã lưu
trong JSON debug. Hãy chạy `python -m pytest` và thử lại ảnh gốc trong môi
trường dự án đã cài đầy đủ dependency trước khi đưa lên production.
