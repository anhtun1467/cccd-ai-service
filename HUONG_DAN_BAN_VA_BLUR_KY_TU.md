# Bản vá OCR ảnh hơi mờ và lỗi ký tự

## Cài đặt

Giải nén gói tại thư mục gốc `cccd-ai-service` và cho phép ghi đè các
tệp cũ.

```powershell
cd "C:\Users\Ahn Tuaann\cccd-ai-service"
Expand-Archive -Force .\cccd_ocr_blur_character_safety_fix.zip .
```

## Hành vi mới

- Không từ chối ảnh chỉ vì điểm Laplacian thấp.
- Ảnh hơi mờ được OCR thêm trên ảnh tăng cường và được giữ khi đọc đủ
  ít nhất ba trường cốt lõi.
- Ảnh thật sự không đọc được trả `OCR_UNREADABLE_IMAGE` thay cho kết luận
  sai dựa trên chuỗi rác.
- `metadata.imageQuality.decision` là `PASSED_WITH_WARNING` khi ảnh hơi mờ
  nhưng vẫn đủ bằng chứng OCR.
- `metadata.validation.fieldValidity` cho biết độ hợp lệ của từng trường.
  Kết quả thiếu địa chỉ hoặc ngày hết hạn không còn được báo `isValid: true`.
- OCR ngày, họ tên và địa chỉ thử cả ảnh xử lý lẫn ảnh raw rồi chọn nguồn
  có cấu trúc tốt hơn.

## Kiểm tra

```powershell
python -m py_compile app\api\ocr.py `
  app\services\ocr_pipeline.py `
  app\modules\ocr\field_ocr_service.py `
  app\modules\ocr\result_fuser.py `
  app\modules\ocr\validator.py

python -m pytest tests\ocr\test_result_fuser.py `
  tests\ocr\test_pipeline_result_fuser.py `
  tests\ocr\test_text_normalizer.py `
  tests\ocr\test_validator.py `
  tests\ocr\test_line_merger.py `
  tests\card_detection\test_geometry_normalization.py `
  tests\test_reject_image.py -v
```

Khởi động lại API:

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
