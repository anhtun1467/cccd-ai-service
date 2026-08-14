# Bản vá OCR ảnh CCCD nghiêng/xéo – 14/08/2026

## Vấn đề đã xử lý

Ảnh CCCD chụp nghiêng hoặc thẻ nằm sát mép ảnh có thể gặp hai lỗi liên
quan nhưng khác nhau:

1. contour bám vào phần bo góc/viền bên trong, sau perspective transform
   phần chữ ở mép phải hoặc mép dưới bị cắt;
2. bốn góc đã được nắn nhưng baseline chữ vẫn dốc nhẹ, làm EasyOCR giảm
   confidence và crop field không trùng hàng chữ.

## Cách xử lý mới

- Giữ phương án perspective transform hiện tại làm phương án chính.
- Sinh thêm tối đa ba phương án hình học khi có đủ bằng chứng:
  `full_frame`, `expanded_contour`, `rotated_rectangle`.
- Chỉ thử các phương án phụ khi OCR chính yếu hoặc hình học cho thấy mức
  phối cảnh cao; không tăng thời gian xử lý của mọi ảnh một cách vô điều
  kiện.
- Chấm từng phương án bằng trường CCCD hợp lệ, nhãn nhận dạng được, số
  dòng/ký tự có nghĩa và confidence; chỉ thay ảnh chính khi điểm vượt
  biên an toàn.
- Đo góc dòng chữ bằng Hough line và box OCR. Nếu còn xiên trong khoảng
  an toàn, thử vertical shear để làm ngang baseline mà vẫn giữ nguyên tỷ
  lệ rộng/cao của thẻ.
- Tiếp tục dùng thư viện glyph tiếng Việt ở OCR từng trường sau khi hình
  học đã được chọn.

## Debug mới

Mỗi lần xử lý có thể xuất thêm các ảnh sau trong `storage/debug/<id>/`:

- `detector_03_candidate_*.jpg`: phương án hình học do detector sinh;
- `detector_03_ocr_candidate_*.jpg`: phương án thực sự được OCR so sánh;
- `detector_03_geometry_selected.jpg`: hình học cuối cùng được chọn;
- `detector_06_skew_candidate_*.jpg`: phương án hiệu chỉnh xiên nhỏ;
- `detector_06_deskewed.jpg`: ảnh cuối bước deskew.

JSON trả về có thêm dữ liệu chẩn đoán trong:

- `metadata.geometry.selection`;
- `metadata.orientation.geometrySelection`;
- `metadata.orientation.residualSkew`.

Các giá trị quan trọng là `selectedCandidate`, `initialScore`,
`selectedScore`, `estimate.angleDegrees` và
`selectedCorrectionDegrees`.

## Cài đặt và chạy

Giải nén gói vá vào thư mục gốc dự án, ghi đè các tệp trùng tên, sau đó:

```bash
pip install -r requirements.txt
pytest -q tests/card_detection
```

Khởi động lại API như cách dự án đang sử dụng. Không cần đổi endpoint hay
request từ frontend.

## Kiểm tra ảnh thực tế

Nên dùng lại cùng bộ ảnh trước/sau và thống kê riêng theo các nhóm góc
chụp 0–5°, 5–10°, 10–20° và trên 20°. Bản vá không nhận bừa phương án
mới: nếu điểm OCR không tốt hơn đủ biên, pipeline sẽ giữ kết quả cũ.
