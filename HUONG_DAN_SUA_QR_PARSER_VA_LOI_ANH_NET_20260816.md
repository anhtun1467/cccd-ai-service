# Sửa QR, parser và lỗi nhận nhầm ảnh nét

## Kết luận từ ảnh kiểm tra

- Ảnh đầu vào có kích thước `1536 x 2048`, điểm nét Laplacian khoảng
  `519.72`: đây là ảnh nét, không phải lỗi lấy nét.
- Detector cũ chia một CCCD nằm ngang trong ảnh dọc thành hai nửa trên/dưới,
  rồi báo sai `MULTIPLE_CARDS`.
- Một số ảnh chụp xa bị contour sáng dính vào nền hoặc tay cầm, làm vùng
  parser lệch dù chữ trên thẻ vẫn rõ.
- OpenCV có thể khoanh đúng QR CCCD mật độ cao nhưng không giải mã được.
  Bản này ưu tiên ZXing-C++ và giữ OpenCV làm fallback.
- API cũ đổi nhiều lỗi khác nhau thành cùng thông báo “Hình ảnh CCCD không
  đạt yêu cầu”, nên người dùng bị yêu cầu chụp lại dù ảnh đã nét.

## Cài bản sửa

Hai ZIP chẩn đoán tải từ kết quả OCR chỉ dùng để kiểm tra; không chép chúng
vào thư mục source. Giải nén ZIP bản sửa trực tiếp vào thư mục gốc dự án,
nơi đang có `app`, `tests` và `requirements.txt`.

PowerShell:

```powershell
cd C:\duong-dan\toi\cccd-ai-service
Expand-Archive "$env:USERPROFILE\Downloads\cccd_qr_parser_diagnostics_fix_20260816.zip" -DestinationPath . -Force
python -m pip install -r requirements.txt
python -m pytest -q
```

Linux/macOS:

```bash
cd /duong-dan/toi/cccd-ai-service
unzip -o ~/Downloads/cccd_qr_parser_diagnostics_fix_20260816.zip
python -m pip install -r requirements.txt
python -m pytest -q
```

Dependency mới bắt buộc trong gói là `zxing-cpp==2.3.0`.

## Ảnh debug mới

Mỗi request tạo các file sau dưới `storage/debug/<request_id>`:

| File | Ý nghĩa |
|---|---|
| `detector_03_geometry_selected.jpg` | Vùng thẻ cuối cùng được chọn sau khi so sánh contour/Hough |
| `qr/qr_01_detection.jpg` | Khoanh QR trên ảnh thẻ: xanh là giải mã thành công, cam là đã thấy nhưng chưa giải mã |
| `qr/qr_02_crop.jpg` | Crop QR dùng để đối chiếu độ nét và vị trí |
| `fields/fields_debug.jpg` | Các vùng parser/OCR từng trường |
| `fields/fields_parser_qr_debug.jpg` | Ảnh parser có thêm khung QR để xem tất cả vùng nhận diện cùng lúc |

Nếu không thấy QR, ảnh `qr_01_detection.jpg` sẽ vẽ vùng tìm kiếm màu vàng.
Payload QR thô không được ghi vào log, metadata hay ảnh debug.

## Phân loại lỗi chi tiết

### Lỗi toàn pipeline

| Mã lỗi | Stage | Ý nghĩa | Hướng xử lý |
|---|---|---|---|
| `MULTIPLE_CARDS` | `CARD_COUNT` | Có từ hai vùng thẻ độc lập; một thẻ nằm ngang trong ảnh dọc không còn bị tách đôi | Chỉ chụp một CCCD |
| `CARD_DETECTION_FAILED` | `CARD_DETECTION` | Không tìm được bốn cạnh thẻ hợp lệ | Xem mask và vùng detector trước khi chỉnh ngưỡng |
| `CCCD_BACK_SIDE_DETECTED` | `CARD_SIDE_CLASSIFICATION` | Ảnh là mặt sau, không phải ảnh mờ | Chụp mặt trước có chân dung và số định danh |
| `OCR_CORE_FIELDS_MISSING` | `PARSER_VALIDATION` | Ảnh đạt chất lượng nhưng parser/vùng cắt chưa xác nhận đủ trường | Xem ảnh geometry và `fields_parser_qr_debug.jpg` |
| `OCR_CORE_FIELDS_MISSING_LOW_QUALITY` | `PARSER_VALIDATION` | Vùng thẻ thật sự tối/mờ và thiếu trường cốt lõi | Chụp lại đủ sáng, giữ máy ổn định |

Với ảnh nét nhưng parser lỗi, `metadata.imageQuality.decision` là
`PASSED_IMAGE_FAILED_PARSER`; API không còn khuyên lấy nét lại.

### Trạng thái QR

| Trạng thái | Ý nghĩa |
|---|---|
| `DECODED_VALID` | Đã khoanh và parse QR CCCD hợp lệ |
| `DETECTED_NOT_DECODED` | Đã khoanh QR nhưng chưa giải mã; OCR chữ vẫn chạy |
| `DECODED_NON_CCCD` | Đọc được payload nhưng không đúng cấu trúc CCCD |
| `NOT_DETECTED` | Không tìm thấy vùng QR; OCR chữ vẫn chạy |
| `DISABLED` | Fast path QR bị tắt trong cấu hình |
| `ERROR` | Nhánh QR lỗi nội bộ; pipeline chuyển sang OCR chữ |

Chi tiết QR nằm trong `metadata.qrFastPath`:

```json
{
  "status": "DECODED_VALID",
  "message": "Đã khoanh và giải mã QR CCCD thành công.",
  "decoded": true,
  "selectedDecoder": "ZXing-C++",
  "regionDetected": true,
  "polygon": [[0, 0], [0, 0], [0, 0], [0, 0]],
  "boundingBox": {"x": 0, "y": 0, "width": 0, "height": 0},
  "providedFields": ["idNumber", "fullName", "dateOfBirth"],
  "errors": [],
  "errorDetails": [],
  "debug": {
    "detectionImage": "storage/debug/<request_id>/qr/qr_01_detection.jpg",
    "cropImage": "storage/debug/<request_id>/qr/qr_02_crop.jpg",
    "parserOverlay": "storage/debug/<request_id>/fields/fields_parser_qr_debug.jpg"
  }
}
```

Các mã QR chi tiết gồm `QR_REGION_NOT_DETECTED`,
`QR_REGION_DETECTED_NOT_DECODED`, `QR_FIELD_COUNT_UNSUPPORTED`,
`QR_ID_NUMBER_INVALID`, `QR_FULL_NAME_INVALID`,
`QR_DATE_OF_BIRTH_INVALID`, `QR_GENDER_INVALID`,
`NON_CCCD_QR_IGNORED`, `QR_TIME_BUDGET_REACHED` và
`QR_FAST_PATH_ERROR`. Mỗi phần tử `errorDetails` có `code`, `stage`,
`message` tiếng Việt và `retryable`.

## Chẩn đoán parser

`metadata.parserDiagnostics` không lặp lại giá trị nhạy cảm; mỗi trường chỉ
ghi trạng thái có/không, hợp lệ, nguồn dữ liệu và độ tin cậy. Khi lỗi, API
trả thêm:

```json
{
  "error_code": "OCR_CORE_FIELDS_MISSING",
  "stage": "PARSER_VALIDATION",
  "reason": "Ảnh đủ nét/sáng nhưng parser chưa xác nhận được đủ trường",
  "image_quality": {
    "decision": "PASSED_IMAGE_FAILED_PARSER"
  },
  "detected_fields": ["idNumber", "fullName"],
  "missing_fields": ["dateOfBirth"],
  "validation_errors": [],
  "qr": {},
  "parser": {},
  "debug_dir": "storage/debug/<request_id>",
  "debug_images": {}
}
```

## Kiểm thử hồi quy

Bản đóng gói đã chạy toàn bộ test hiện tại: `203 passed, 1 skipped`.
Các ca mới bao gồm một thẻ nằm ngang trong ảnh dọc, chọn Hough khi contour
dính nền, khoanh/crop QR, parser QR lỗi chi tiết, nhận diện mặt trước/mặt
sau và thông báo parser không đổ lỗi nhầm cho độ nét.
