# Bản vá OCR CCCD: dấu tiếng Việt, xoay ảnh và hiệu chỉnh phối cảnh

Giải nén gói ZIP vào thư mục gốc `cccd-ai-service` và chọn ghi đè.

## Nội dung đã sửa

- EasyOCR dùng `vi + en`, Beam Search và allowlist đầy đủ ký tự tiếng Việt.
- Không xóa nhầm các chữ `Đ`, `Ư`, `Ơ`, `Ạ`, `Ữ` trong Field OCR.
- Ưu tiên nguồn OCR có dấu và phục hồi có kiểm soát họ tên/địa danh.
- Giá trị quốc tịch chuẩn hóa thành `Việt Nam`.
- Dùng bốn góc thật của CCCD để hiệu chỉnh phối cảnh, không dùng bounding box ngang.
- Tự đưa ảnh dọc về ngang; khi nghi ngờ ảnh úp ngược, thử 180 độ và chọn chiều OCR tốt hơn.
- Lưu ảnh cuối cùng đã đúng chiều tại `storage/debug/<request_id>/detector_05_oriented.jpg`.
- Trả thêm `metadata.geometry` và `metadata.orientation` trong JSON.

## Kiểm tra

```powershell
python -m py_compile app\modules\card_detection\contour_detector.py `
app\modules\card_detection\perspective_transformer.py `
app\modules\ocr\easyocr_engine.py `
app\modules\ocr\field_ocr_service.py `
app\modules\ocr\result_fuser.py `
app\services\ocr_pipeline.py

python -m pytest tests\ocr\test_result_fuser.py `
tests\ocr\test_pipeline_result_fuser.py `
tests\card_detection\test_geometry_normalization.py -v

python tests\ocr\test_text_normalizer.py
```

## Khởi động lại API

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Lần chạy đầu sau khi bật ngôn ngữ `vi` có thể cần tải mô hình EasyOCR tiếng Việt.
