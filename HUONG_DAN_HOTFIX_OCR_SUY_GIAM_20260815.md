# Hotfix khôi phục khả năng đọc CCCD sau bản adaptive

## Nguyên nhân hồi quy

Bản tăng sáng/khử mờ trước có hai thay đổi quá mạnh:

1. OCR toàn thẻ có thể thay ảnh warped gốc bằng một adaptive variant chỉ
   vì chênh `0.45` điểm. Điểm tăng nhẹ do confidence hoặc nhãn không đảm
   bảo Layout Parser còn đọc được đầy đủ trường.
2. Các crop `low_light` và `deblur` được chèn trước `value_raw`, `binary`
   và các crop ổn định cũ. Một kết quả nhìn có vẻ hợp lệ có thể làm OCR
   dừng sớm trước khi thử ảnh cũ tốt hơn.

## Cách hotfix xử lý

- Khôi phục `detector_03_warped`/ảnh card gốc làm nguồn full-card OCR chính.
- Bỏ adaptive variant khỏi đường chạy OCR field mặc định và khôi phục đúng
  thứ tự crop cũ.
- Không tăng thêm số lượt OCR field.
- Ảnh enhanced chỉ là fallback khi vùng thẻ mờ, tối hoặc OCR chiều ảnh yếu.
- Cấm chọn enhanced nếu nó làm giảm số trường lõi hợp lệ hoặc số dòng có
  nghĩa, kể cả khi confidence tổng cao hơn.
- Ảnh portrait tiếp tục lấy từ card warped gốc, không bị tăng nét hay đổi
  sáng, nên Face Verification không thay đổi.

## Cài đặt

Giải nén gói hotfix vào thư mục gốc `cccd-ai-service` và chọn ghi đè các
tệp trùng tên. Sau đó chạy trong PowerShell:

```powershell
python -m pytest `
  tests/ocr/test_pipeline_result_fuser.py `
  tests/ocr/test_result_fuser.py `
  tests/ocr/test_text_normalizer.py `
  tests/ocr/test_low_light_field_variants.py `
  tests/test_reject_image.py -v
```

Khởi động lại API:

```powershell
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Kiểm tra lại đúng ảnh đã lỗi

Với mỗi ảnh, đối chiếu:

- `storage/debug/<ten-anh>/detector_03_warped.jpg`;
- `storage/outputs/<ten-anh>_card.jpg`;
- `storage/outputs/<ten-anh>_enhanced.jpg`;
- `storage/json/<ten-anh>.json`;
- `metadata.fullCardOcrVariant` phải ưu tiên `card`; chỉ là `enhanced` khi
  enhanced không làm mất trường hoặc dòng.

Nếu một ảnh vẫn đọc kém, gửi đúng thư mục debug và JSON của ảnh đó. Không
điều chỉnh thêm ngưỡng chung từ một ảnh đơn lẻ vì có thể làm hỏng các mẫu
CCCD cũ/Căn cước mới còn lại.
