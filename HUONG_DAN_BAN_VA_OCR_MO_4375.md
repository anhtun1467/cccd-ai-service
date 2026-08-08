# Bản vá OCR ảnh hơi mờ – request 4375

## Cài đặt

Giải nén gói tại thư mục gốc dự án và chọn ghi đè:

```powershell
cd "C:\Users\Ahn Tuaann\cccd-ai-service"
Expand-Archive -Force .\cccd_ocr_blurry_multivariant_4375_fix.zip .
```

## Các thay đổi chính

- Không dùng điểm Laplacian sau warp làm kết luận duy nhất cho ảnh mờ.
- Tạo vùng cắt tập trung vào **giá trị** của từng trường, bên cạnh vùng
  rộng có nhãn như trước.
- Mỗi trường được thử tối đa ba biến thể: màu gốc, CLAHE + sharpen nhẹ và
  nhị phân cho số/ngày; dừng sớm khi đã có kết quả đủ mạnh.
- EasyOCR dùng allowlist riêng cho số CCCD và ngày, đồng thời hạ ngưỡng
  phát hiện chữ ở vùng ảnh mờ.
- Ghép dấu họ tên theo từng từ khi nhiều lần OCR đọc cùng chuỗi không dấu.
- Dùng cấu trúc số CCCD để kiểm tra/sửa **năm sinh** và giới tính; không tự
  tạo ngày/tháng khi ảnh không cung cấp được dữ liệu.
- Ảnh lỗi vẫn lưu raw text, field candidates và dữ liệu đọc được trong
  JSON `_error.json` để lần debug sau không bị mất bằng chứng.
- Chỉ dùng một EasyOCR Reader cho toàn thẻ và field, giảm bộ nhớ/model load.

## Kiểm tra

```powershell
python -m py_compile app\modules\card_detection\enhancer.py `
  app\modules\ocr\easyocr_engine.py `
  app\modules\ocr\field_cropper.py `
  app\modules\ocr\field_ocr_service.py `
  app\modules\ocr\result_fuser.py `
  app\services\ocr_pipeline.py

python -m pytest tests\ocr\test_result_fuser.py `
  tests\ocr\test_pipeline_result_fuser.py `
  tests\ocr\test_text_normalizer.py `
  tests\ocr\test_validator.py `
  tests\ocr\test_line_merger.py `
  tests\ocr\test_blurry_field_variants.py `
  tests\card_detection\test_geometry_normalization.py `
  tests\test_reject_image.py -v
```

Khởi động lại API:

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Ảnh debug vùng giá trị mới được lưu tại:

```text
storage/debug/<request_id>/fields/fields_value_debug.jpg
```

Nếu OCR vẫn không đủ bằng chứng, tệp `_error.json` giờ giữ `cccdData`,
`rawText`, `textBoxes`, `fieldResults` và từng `ocrCandidates` để kiểm tra.
