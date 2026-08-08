# Bản vá nhiều CCCD và lỗi ký tự OCR

## Cài đặt

Giải nén gói vá vào thư mục gốc của dự án và cho phép ghi đè file cũ:

```powershell
cd "C:\Users\Ahn Tuaann\cccd-ai-service"
Expand-Archive -Force .\cccd_ocr_multi_card_character_fix.zip .
```

Kiểm tra cú pháp và test hồi quy:

```powershell
python -m py_compile app\api\ocr.py `
  app\modules\card_detection\contour_detector.py `
  app\modules\card_detection\detector.py `
  app\modules\ocr\easyocr_engine.py `
  app\modules\ocr\field_cropper.py `
  app\modules\ocr\field_ocr_service.py `
  app\modules\ocr\result_fuser.py `
  app\services\ocr_pipeline.py `
  app\utils\image_validator.py

python -m pytest tests\ocr\test_result_fuser.py `
  tests\ocr\test_pipeline_result_fuser.py `
  tests\ocr\test_line_merger.py `
  tests\card_detection\test_geometry_normalization.py `
  tests\test_reject_image.py -v
```

Khởi động lại API:

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Hành vi mới

- Ảnh chứa hai CCCD không còn bị báo nhầm là ảnh mờ. API trả HTTP 400 với
  `error_code = MULTIPLE_CARDS` và `card_count = 2`.
- Ảnh đầu vào chỉ bị chặn sớm khi rất mờ hoặc quá tối. Sau khi phát hiện và
  làm phẳng thẻ, vùng CCCD được kiểm tra lại bằng ngưỡng chặt hơn.
- Ảnh một thẻ nhưng chữ thực sự quá mờ trả `CARD_BLURRY_IMAGE`, không tạo dữ
  liệu OCR sai.
- EasyOCR dùng mô hình `vi + en`, đầy đủ bảng chữ cái tiếng Việt và decoder
  `greedy` để tránh cảnh báo overflow của beam search.
- Các vùng họ tên, số CCCD, quê quán và nơi thường trú được mở rộng; các
  trường quan trọng được OCR lại cả ảnh đã xử lý và ảnh raw rồi chọn kết quả
  tốt hơn.
- Bộ hợp nhất sửa các lỗi nhãn, dấu và ký tự đã gặp trong bộ JSON debug,
  nhưng không tự đoán dữ liệu khi ảnh nguồn không đủ nét.

Với ảnh có hai CCCD, kết quả lỗi mong đợi có dạng:

```json
{
  "detail": {
    "message": "Hình ảnh CCCD không đạt yêu cầu.",
    "error_code": "MULTIPLE_CARDS",
    "reason": "Phát hiện nhiều CCCD trong cùng một ảnh",
    "card_count": 2,
    "suggestion": "Vui lòng chỉ chụp một CCCD trong mỗi ảnh."
  }
}
```
