# Tối ưu ảnh phóng lớn, ảnh dọc và ảnh ngược — 18/08/2026

Gói này là bản **tích lũy**, thay thế các ZIP QR/parser/crop trước đó. Giải nén
thẳng vào thư mục gốc dự án, nơi đang có `app`, `tests` và `requirements.txt`.

## Các lỗi đã xử lý

1. **Ảnh dọc hoặc xoay ngược chạy lâu**
   - Nhận chiều 0/180 độ trước EasyOCR bằng ba tín hiệu nhẹ: vị trí QR, finder
     pattern QR và bố cục đỏ của quốc huy/tiêu đề mặt trước.
   - Vùng QR vuông đúng góc vẫn có thể xác định chiều dù payload chưa giải mã.
   - Nếu tín hiệu không đủ chắc chắn, pipeline tự quay về cơ chế OCR thử chiều
     cũ; không tự xoay theo dự đoán yếu.
   - Không giải mã QR lần hai chỉ vì đã xoay 180 độ. Tọa độ khoanh QR được đổi
     trực tiếp sang ảnh đã xoay.

2. **Ảnh phóng lớn bị cắt mất dòng sát mép**
   - Contour có nội dung CCCD phủ gần toàn khung được khóa là toàn thẻ, không
     còn bị Hough thay bằng QR, chân dung hoặc một dải chữ nội bộ.
   - Kết quả geometry ghi rõ `frameEdgesTouched` và
     `cardPossiblyClippedByFrame` để phân biệt crop sai với ảnh nguồn đã mất mép.
   - Ảnh warped có cạnh dài tối đa 1.800 px. Mức này vẫn cao hơn nhiều so với
     ảnh chuẩn hóa field 1000×630, nhưng giảm mạnh thời gian warp/JPEG/EasyOCR
     của ảnh camera 3K–6K.

3. **Lặp OCR/QR không cần thiết**
   - OCR ảnh enhanced chỉ chạy khi OCR chính yếu và chưa có QR hợp lệ.
   - QR cuối pipeline chỉ chạy lại nếu geometry hoặc deskew thật sự thay đổi.
   - Metadata cho biết rõ `retryAttempted`, `retryReason` hoặc
     `retrySkippedReason`.

4. **Ảnh rõ nhưng báo chất lượng thấp**
   - Điểm Laplacian dưới ngưỡng thường chỉ là cảnh báo, không còn tự tăng số
     trường parser bắt buộc.
   - Chỉ ảnh có blur dưới 28 hoặc độ sáng dưới 30 mới được xếp vào mức chất
     lượng nghiêm trọng.
   - Khi ảnh vẫn đọc thiếu trường, lỗi được gắn đúng stage
     `PARSER_VALIDATION`; không đổ thành lỗi chất lượng chỉ vì cảnh báo mềm.

## Cài vào dự án Windows

Mở PowerShell:

```powershell
cd "C:\Users\Ahn Tuaann"
Copy-Item ".\cccd-ai-service" ".\cccd-ai-service_backup_20260818" -Recurse
Expand-Archive `
  -Path ".\Downloads\cccd_zoom_portrait_upside_down_speed_fix_20260818.zip" `
  -DestinationPath ".\cccd-ai-service" `
  -Force
cd ".\cccd-ai-service"
pip install -r requirements.txt
pytest -q
```

Sau đó khởi động lại API bằng lệnh dự án đang dùng. Không đặt ZIP vào trong
`app`, `app/modules/qr` hay `storage`; nội dung ZIP đã có đúng cấu trúc thư mục.

## Kiểm tra kết quả

Trong JSON, kiểm tra:

- `metadata.processingStagesMs.fastVisualOrientation`
- `metadata.orientation.orientationSource`
- `metadata.orientation.orientationRetried`
- `metadata.qrFastPath.retryAttempted`
- `metadata.geometry.outputScaleLimited`
- `metadata.geometry.cardPossiblyClippedByFrame`
- `metadata.imageQuality.decision`
- `metadata.parserDiagnostics.fields`

Ảnh debug quan trọng:

- `storage/debug/<request-id>/detector_03_geometry_selected.jpg`
- `storage/debug/<request-id>/detector_05_oriented.jpg`
- `storage/debug/<request-id>/qr/qr_01_detection.jpg`
- `storage/debug/<request-id>/fields/fields_values_parser_qr_debug.jpg`

Ảnh cuối phải là thẻ ngang, đúng chiều; khung QR nằm góc trên-phải và các vùng
giá trị không mất dòng sát đáy.

## Benchmark không lộ dữ liệu CCCD

```powershell
python scripts\benchmark_orientation_zoom.py "C:\duong-dan\anh-test.jpg" --loops 20
```

Script chỉ in kích thước, chiều, trạng thái QR và thời gian; không in payload QR
hoặc nội dung các trường CCCD.

## Kết quả xác minh của gói

- `216 passed, 1 skipped`.
- Ảnh mẫu được thử ở các chiều 0°, 90°, 180° và 270°: cả bốn đều xác định được
  chiều trước OCR.
- Bộ nhận chiều nhẹ có trung vị khoảng 6,8 ms trên ảnh thẻ mẫu; QR hợp lệ được
  giải mã ngay lượt đầu trong phép đo nội bộ.

Lưu ý: nếu ảnh nguồn thực sự đã cắt mất một phần thẻ thì phần pixel đó không thể
khôi phục bằng phần mềm. Khi `cardPossiblyClippedByFrame=true` và ảnh
`detector_03_geometry_selected.jpg` cũng thiếu mép, cần chụp lại trọn thẻ.
