# Tối ưu QR, parser, crop thẻ, tốc độ và đánh giá chất lượng ảnh

## Kết luận từ 6 lần chạy được cung cấp

Phân tích được thực hiện trên metadata thời gian và ảnh debug, không đưa ảnh
CCCD, payload QR thô hoặc dữ liệu cá nhân vào gói bàn giao.

| Hiện tượng | Nguyên nhân xác định | Sửa trong bản này |
|---|---|---|
| QR rõ nhưng không nhận | ZXing chỉ thử toàn thẻ một lần; OpenCV không ổn định với QR CCCD mật độ cao | ZXing thử toàn thẻ, ROI trên-phải và ROI dưới-trái đã CLAHE/sharpen; hỗ trợ xoay 180° và ánh xạ khung QR về ảnh thẻ |
| Parser sai/thiếu trường | Crop sai ở một ca; field OCR chưa tận dụng đủ dữ liệu đã đọc từ toàn thẻ và QR | Sửa crop toàn thẻ, hợp nhất sơ bộ làm reference, ưu tiên dữ liệu QR hợp lệ và vẫn giữ nguồn dữ liệu từng field |
| Khoanh/cắt lệch | Hough chọn một dải chữ nội bộ thay vì toàn bộ thẻ khi contour chạm biên có tỷ lệ méo | Thêm fallback `large_foreground_contour` dựa trên diện tích, rectangularity và phân bố nội dung toàn thẻ; bỏ Hough khi contour này đã đáng tin cậy |
| Chậm | QR chạy sau OCR; orientation và residual-skew có thể gọi EasyOCR lại; field OCR thử nhiều biến thể | QR chạy trước OCR để xác định chiều; tái sử dụng kết quả QR; bỏ OCR deskew khi QR + OCR ban đầu đã mạnh; dùng reference để dừng field OCR sớm |
| Ảnh nét vẫn bị báo lỗi | Trạng thái parser/QR cần xem lại bị gán vào `imageQuality.decision` | `imageQuality.decision` chỉ phản ánh nét/sáng; lỗi parser và xung đột dữ liệu nằm ở `validation`/`reviewRequired` |

Ba kết quả trước đây là `OCR_PARTIAL` dù đủ 8 trường chỉ vì địa chỉ QR khác
chuỗi OCR. Địa chỉ QR hợp lệ nay là nguồn chính; khác biệt OCR được ghi trong
`qrAdvisoryDifferenceFields` để chẩn đoán nhưng không làm hỏng toàn kết quả.
Xung đột trường định danh, họ tên, ngày sinh hoặc giới tính vẫn yêu cầu kiểm
tra như trước.

## Cài đặt trên Windows

Giải nén ZIP này vào **thư mục gốc dự án**, tức thư mục đang có `app`, `tests`
và `requirements.txt`. Không chép các ZIP/JSON debug của từng request vào mã
nguồn.

PowerShell:

```powershell
cd "C:\Users\Ahn Tuaann\cccd-ai-service"
Expand-Archive "$env:USERPROFILE\Downloads\cccd_qr_parser_crop_speed_optimized_20260816.zip" -DestinationPath . -Force
python -m pip install -r requirements.txt
python -m pytest -q
```

Dependency QR cần có là `zxing-cpp==2.3.0`. Sau khi cài, khởi động lại API để
Python nạp mã mới.

## Ảnh debug cần xem

Mỗi request tạo các ảnh dưới `storage/debug/<request_id>`:

| File | Dùng để kiểm tra |
|---|---|
| `detector_03_geometry_selected.jpg` | Toàn bộ thẻ cuối cùng được chọn, không phải dải chữ nội bộ |
| `qr/qr_01_detection.jpg` | Polygon QR thật; khi chưa thấy sẽ khoanh cả hai vùng tìm kiếm trên-phải/dưới-trái |
| `qr/qr_02_crop.jpg` | Crop QR đã phát hiện, không chứa payload dạng chữ |
| `fields/fields_parser_qr_debug.jpg` | Khung field rộng của parser cùng khung QR |
| `fields/fields_values_parser_qr_debug.jpg` | Khung **giá trị** hẹp hơn của từng field cùng khung QR; nên dùng ảnh này để chỉnh lệch field |

`metadata.qrFastPath` có thêm `searchRegions`,
`orientationRotationDegrees`, `orientationProbeAttemptCount` và
`orientationProbeElapsedMs`. Payload QR thô không được trả về API, ghi log
hoặc vẽ lên ảnh debug.

## Kết quả kiểm chứng

- Toàn bộ test: `208 passed, 1 skipped`.
- Mẫu QR trước đây không nhận: 20/20 lượt ngay trên ảnh detector đang xoay
  180° và 20/20 lượt sau khi QR đã chuẩn hóa chiều.
- Thời gian QR trên mẫu đó: trung vị khoảng 13.65 ms trước khi xoay và
  10.76 ms sau khi xoay; p95 lần lượt khoảng 23.00 ms và 11.37 ms.
- Ca crop sai trước đây chọn quad nội bộ có diện tích khoảng 21.9% khung. Bản
  mới chọn toàn thẻ khoảng 65.76% khung và đánh dấu
  `LARGE_FOREGROUND_CONTOUR_COMPLETE`.
- Các lượt residual-skew OCR trong log cũ tốn khoảng 7.8-17.5 giây nhưng không
  cần thiết khi QR và OCR ban đầu đã đủ mạnh; nhánh này nay được bỏ qua với
  `skipReason=QR_AND_STRONG_OCR`.

Thời gian tổng thực tế còn phụ thuộc CPU/GPU và EasyOCR trên máy triển khai.
Hãy so sánh `metadata.processingStagesMs`, đặc biệt các mục
`qrOrientationProbe`, `orientation`, `residualSkew` và `fieldCropAndOcr`.

## Ý nghĩa trạng thái sau sửa

- `imageQuality.decision=PASSED`: ảnh đủ nét/sáng, kể cả khi parser có trường
  cần xem lại.
- `imageQuality.decision=PASSED_WITH_WARNING`: điểm nét hoặc sáng thấp, nhưng
  pipeline vẫn đọc được đủ trường cốt lõi.
- `validation.qrConflictFields`: xung đột quan trọng cần kiểm tra.
- `validation.qrAdvisoryDifferenceFields`: khác biệt tham khảo, hiện dùng cho
  địa chỉ QR so với OCR và không ép `OCR_PARTIAL`.
- `OCR_CORE_FIELDS_MISSING`: ảnh đủ chất lượng nhưng crop/parser chưa đọc đủ;
  kiểm tra ảnh geometry và ảnh khoanh value trước khi yêu cầu chụp lại.
- `OCR_CORE_FIELDS_MISSING_LOW_QUALITY`: vừa thiếu trường cốt lõi vừa có bằng
  chứng ảnh tối/mờ thực sự.
