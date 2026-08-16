# Bản sửa cắt CCCD chụp xa và tách hai trường địa chỉ

## Nội dung đã sửa

- Card Detection có thêm fallback tìm hai cặp cạnh song song bằng Hough khi
  threshold vùng sáng làm thẻ dính vào bàn tay, màn hình hoặc mặt bàn.
- Bốn góc được nắn về đúng tỷ lệ vật lý ID-1 `85.60 / 53.98`.
- Chỉ dùng ảnh thu nhỏ để tìm góc. Nếu ảnh đầu vào lớn hơn, hệ thống chiếu
  góc về ảnh gốc và cắt từ ảnh gốc để không mất nét chữ ở ảnh chụp xa.
- Ảnh vốn đã có contour tốt vẫn dùng contour cũ làm kết quả chính; Hough chỉ
  là ứng viên OCR dự phòng, tránh thay một crop đang đúng bằng vùng nội bộ.
- `Quê quán` và `Nơi thường trú` dùng một ranh giới Y chung nên không chồng
  lấn. Khi OCR toàn thẻ nhận được nhãn nơi thường trú, ranh giới tự bám theo
  mép trên của nhãn đó.
- Vùng địa chỉ bắt đầu từ X=300 trên ảnh chuẩn `1000 x 630`, giảm lấy nhầm
  ảnh chân dung và cột ngày hết hạn ở bên trái.

## Cài bản sửa

Giải nén gói tại đúng thư mục gốc dự án, nơi đang có `app`, `tests` và
`requirements.txt`:

```powershell
Expand-Archive -Path .\cccd_ocr_far_shot_address_crop_fix_20260816.zip `
  -DestinationPath . -Force
```

Không chép thư mục `venv`, model OCR, ảnh CCCD thật, `storage/debug` hoặc
`storage/outputs` vào Git.

## Kiểm tra nhanh

```powershell
python -m pytest -q `
  tests/card_detection/test_far_shot_card_crop.py `
  tests/ocr/test_address_crop_boundaries.py `
  tests/ocr/test_spatial_field_alignment.py
```

Khi gọi API với `debug` đang bật, kiểm tra:

- `detector_03_warped.jpg`: chỉ còn thẻ, đủ bốn mép, không còn nền.
- `fields/fields_debug.jpg`: đáy `placeOfOrigin` trùng chính xác với đỉnh
  `placeOfResidence`.
- JSON `metadata.geometry.geometrySource` là `original_image` đối với ảnh
  đầu vào lớn hơn ảnh dùng để detect.
- JSON `metadata.orientation.addressCropLayout.source` là
  `residence_label` khi OCR nhận được nhãn nơi thường trú.

## Lưu ý hồi quy

Phần chân dung vẫn được lấy từ ảnh thẻ đã nắn nhưng chưa qua các biến thể
low-light/deblur. Bản sửa không thay đổi ngưỡng Face Verification và không
đưa lại cơ chế enhancement toàn thẻ từng làm giảm khả năng đọc sau cập nhật.

