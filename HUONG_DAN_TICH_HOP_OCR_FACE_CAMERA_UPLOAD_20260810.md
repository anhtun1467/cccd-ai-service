# Hướng dẫn tích hợp OCR → Face Verification → Camera/Upload

## Mở đúng giao diện camera

Swagger `/docs` là trang thử API bằng file upload; ô
`capture_source=camera` trong Swagger không tự bật webcam. Sau bản vá giao
diện, có ba địa chỉ tương đương:

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/camera
http://127.0.0.1:8000/face-verification
```

Trang sẽ hiện nút **📷 MỞ CAMERA NGAY** ở đầu màn hình. Bấm nút này để cuộn
tới khu vực webcam và xin quyền camera ngay. Trong Swagger cũng có liên kết
**MỞ GIAO DIỆN CAMERA TRỰC TIẾP** ở phần mô tả đầu trang.

Nếu Chrome chưa hiện hình camera:

1. Chỉ dùng `127.0.0.1` hoặc `localhost`, không dùng `0.0.0.0` trên thanh địa
   chỉ.
2. Bấm biểu tượng camera ở thanh địa chỉ → **Cho phép**.
3. Tắt Windows Camera, OBS, Zoom, Discord hoặc ứng dụng khác đang giữ webcam.
4. Bấm `Ctrl + F5`, chọn Logitech C922 rồi bấm **MỞ CAMERA** lại.
5. Kiểm tra quyền Windows tại **Settings → Privacy & security → Camera** và
   bật quyền cho ứng dụng desktop.

## 1. Kết quả sau bản cập nhật

Luồng mới chỉ gửi CCCD đúng một lần:

1. Frontend gửi ảnh mặt trước CCCD vào `POST /ocr/cccd`.
2. OCR phát hiện thẻ, làm phẳng, đọc dữ liệu và lưu ảnh tham chiếu ở server.
3. Response OCR trả thêm `face_session.session_id`.
4. Người dùng chọn một trong hai cách tạo selfie:
   - Mở camera trình duyệt và chụp ảnh.
   - Tải ảnh selfie JPG/JPEG/PNG lên.
5. Frontend chỉ gửi `session_id` và `selfie_image` vào
   `POST /api/face-verification/verify-from-ocr`.
6. Server tự lấy crop chân dung của OCR. Nếu crop khó phát hiện mặt, server
   tự dùng ảnh CCCD đã làm phẳng; người dùng không phải tải CCCD lại.

Endpoint cũ `POST /api/face-verification/verify` vẫn tồn tại để mã cũ không
bị lỗi, nhưng đã được đánh dấu `deprecated` trong Swagger.

## 2. Cấu trúc bảo mật của phiên

- `session_id` được tạo bằng bộ sinh số ngẫu nhiên bảo mật, không chứa số CCCD.
- Payload `face_session` và endpoint Face không trả/nhận
  `card_image_path` hoặc `portrait_image_path`. Các trường đường dẫn debug vốn
  có trong response OCR cũ được giữ nguyên để không làm hỏng công cụ debug.
- Mọi đường dẫn ảnh được kiểm tra phải nằm trong `storage/outputs` hoặc
  `storage/debug`; client không thể truyền đường dẫn tùy ý.
- Phiên mặc định hết hạn sau 30 phút.
- Mỗi phiên có tối đa 5 lần đưa ảnh qua model.
- Khi `MATCH`, phiên chuyển sang `verified` và không dùng lại được.
- Khi hết số lần thử, phiên chuyển sang `exhausted`.
- Bản ghi phiên hết hạn được dọn sau 24 giờ; việc dọn bản ghi không xóa ảnh
  OCR đang được dự án dùng để debug.
- Selfie và crop Face không được lưu nếu `FACE_SAVE_DEBUG=False` (mặc định).
- `capture_source=camera` chỉ là metadata UX. Một ảnh tĩnh không phải bằng
  chứng liveness; response luôn trả `liveness_checked=false`.

Có thể thay đổi trong `.env`:

```dotenv
FACE_SESSION_DIR=storage/face_sessions
FACE_SESSION_TTL_SECONDS=1800
FACE_SESSION_MAX_ATTEMPTS=5
FACE_SESSION_EXPIRED_RETENTION_SECONDS=86400
FACE_SESSION_LEASE_TIMEOUT_SECONDS=300
FACE_SAVE_DEBUG=False
```

## 3. Cài đặt

Tại PowerShell trong thư mục dự án:

```powershell
cd "C:\Users\Ahn Tuaann\cccd-ai-service"

python -m pip install --only-binary=:all: -r requirements-face.txt
python -m pip check

python -c "import insightface, onnxruntime as ort; print('InsightFace:', insightface.__version__); print('ONNX Runtime:', ort.__version__); print('Providers:', ort.get_available_providers())"
```

Kết quả cần có:

```text
InsightFace: 1.0.1
ONNX Runtime: 1.27.0
CPUExecutionProvider
```

Không hạ về `insightface==0.7.3` trên Python 3.11/Windows.

## 4. Khởi động và mở giao diện

```powershell
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Mở:

- Giao diện OCR + camera/upload:
  `http://localhost:8000/face-verification`
- Swagger:
  `http://localhost:8000/docs`

Camera trình duyệt hoạt động trên `localhost`, `127.0.0.1` hoặc HTTPS. Nếu
truy cập qua IP LAN bằng HTTP, Chrome có thể chặn `getUserMedia()`.

## 5. API OCR và phiên Face

### Request OCR

```powershell
curl.exe -X POST "http://localhost:8000/ocr/cccd" `
  -H "accept: application/json" `
  -F "file=@C:\duong-dan\mat-truoc-cccd.jpg;type=image/jpeg"
```

Response được bổ sung phần sau (rút gọn):

```json
{
  "success": true,
  "data": {
    "status": "OCR_SUCCESS",
    "cccdData": {
      "idNumber": "...",
      "fullName": "..."
    },
    "face_session": {
      "session_id": "opaque-random-session-id",
      "ocr_request_id": "uuid-cua-anh-ocr",
      "status": "active",
      "created_at": "2026-08-10T00:00:00Z",
      "expires_at": "2026-08-10T00:30:00Z",
      "max_attempts": 5,
      "attempts_used": 0,
      "remaining_attempts": 5,
      "can_verify": true,
      "verify_endpoint": "/api/face-verification/verify-from-ocr"
    }
  }
}
```

`OCR_PARTIAL` vẫn có thể tạo phiên Face nếu ảnh thẻ đã được phát hiện và ảnh
tham chiếu còn tồn tại. `OCR_FAILED` không tạo phiên.

## 6. API Face từ OCR

### Selfie upload

```powershell
$sessionId = "SESSION_ID_NHAN_TU_OCR"

curl.exe -X POST `
  "http://localhost:8000/api/face-verification/verify-from-ocr" `
  -H "accept: application/json" `
  -F "session_id=$sessionId" `
  -F "capture_source=upload" `
  -F "selfie_image=@C:\duong-dan\selfie.jpg;type=image/jpeg"
```

Ảnh chụp từ camera gửi cùng endpoint, chỉ đổi:

```text
capture_source=camera
```

Không có field `card_image` trong endpoint mới.

### Response thành công

```json
{
  "success": true,
  "request_id": "20260810_000000_ab12cd34",
  "ocr_session_id": "opaque-random-session-id",
  "capture_source": "camera",
  "status": "match",
  "is_match": true,
  "needs_review": false,
  "similarity": 0.7312,
  "match_threshold": 0.5,
  "review_threshold": 0.4,
  "portrait_method": "ocr_portrait_crop",
  "reference_source": "ocr_portrait_crop",
  "embedding_dimension": 512,
  "model_name": "buffalo_l",
  "liveness_checked": false,
  "cccd_quality": {},
  "webcam_quality": {},
  "session": {
    "status": "verified",
    "attempts_used": 1,
    "remaining_attempts": 4,
    "can_verify": false
  }
}
```

`reference_source` có thể là:

- `ocr_portrait_crop`: dùng trực tiếp crop chân dung do OCR tạo.
- `ocr_card_image_fallback`: crop OCR không đủ tốt, tự phát hiện lại trên ảnh
  thẻ đã làm phẳng.
- `ocr_card_image`: OCR không có crop portrait, dùng ảnh thẻ đã làm phẳng.
- `uploaded_card`: chỉ xuất hiện ở endpoint tương thích cũ.

## 7. Trạng thái so khớp

- `match`: similarity `>= 0.50` và chất lượng ảnh không buộc hạ mức.
- `review`: similarity từ `0.40` đến `< 0.50`, hoặc đạt ngưỡng match nhưng
  chất lượng ảnh có cảnh báo.
- `not_match`: similarity `< 0.40`.

Không coi `review` là trùng khớp tự động. Nên cho phép chụp lại; nếu vẫn
`review`, chuyển sang kiểm tra thủ công.

## 8. Quản lý phiên

Kiểm tra trạng thái:

```powershell
curl.exe "http://localhost:8000/api/face-verification/sessions/$sessionId"
```

Hủy phiên:

```powershell
curl.exe -X DELETE `
  "http://localhost:8000/api/face-verification/sessions/$sessionId"
```

## 9. Mã lỗi chính

| HTTP | `errorCode` | Ý nghĩa / xử lý |
|---:|---|---|
| 400 | `INVALID_FACE_SESSION_ID` | Session id sai định dạng. |
| 400 | `INVALID_SELFIE_IMAGE` | File rỗng, hỏng hoặc quá lớn. |
| 404 | `FACE_SESSION_NOT_FOUND` | Không có phiên; chạy OCR lại. |
| 409 | `FACE_SESSION_ALREADY_VERIFIED` | Phiên đã MATCH và bị khóa. |
| 409 | `FACE_SESSION_BUSY` | Một request khác đang dùng cùng phiên; chờ request đó xong. |
| 410 | `FACE_SESSION_EXPIRED` | Quá 30 phút; chạy OCR lại. |
| 410 | `OCR_CARD_IMAGE_MISSING` | Ảnh tham chiếu đã bị xóa; chạy OCR lại. |
| 410 | `OCR_REFERENCE_IMAGE_INVALID` | Ảnh tham chiếu bị hỏng; chạy OCR lại. |
| 415 | `UNSUPPORTED_IMAGE_TYPE` | Chỉ dùng JPG/JPEG/PNG. |
| 422 | `WEBCAM_FACE_NOT_FOUND` | Không thấy mặt; nhìn thẳng và chụp lại. |
| 422 | `MULTIPLE_WEBCAM_FACES` | Ảnh phải có đúng một người. |
| 422 | `WEBCAM_FACE_TOO_BLURRY` | Ảnh quá mờ. |
| 422 | `WEBCAM_FACE_TOO_DARK` | Ảnh quá tối. |
| 422 | `WEBCAM_FACE_TOO_BRIGHT` | Ảnh bị cháy sáng. |
| 422 | `WEBCAM_FACE_TOO_SMALL` | Đứng gần camera hơn. |
| 429 | `FACE_SESSION_ATTEMPTS_EXHAUSTED` | Đã hết số lần thử; OCR lại. |
| 503 | `FACE_MODEL_UNAVAILABLE` | Kiểm tra model `buffalo_l` và ONNX Runtime. |
| 503 | `FACE_MODEL_INFERENCE_FAILED` | Kiểm tra provider/model/runtime. |

Lỗi chất lượng hoặc không phát hiện mặt được tính là một lần thử. Lỗi model
`503` không bị tính vào số lần thử.

## 10. Test

```powershell
python -m pytest tests\face_verification -v
python -m pytest
```

Test camera thật Logitech C922:

```powershell
$env:RUN_CAMERA_TEST="1"
python -m pytest tests\face_verification\test_camera_source.py -v
Remove-Item Env:RUN_CAMERA_TEST
```

Khi không đặt `RUN_CAMERA_TEST=1`, một test camera bị `SKIPPED` là bình thường.

## 11. Cách dùng trong Frontend React

Sau khi OCR:

```javascript
const ocrForm = new FormData();
ocrForm.append("file", cccdFile);

const ocrResponse = await fetch("http://localhost:8000/ocr/cccd", {
  method: "POST",
  body: ocrForm,
});
const ocrPayload = await ocrResponse.json();
const sessionId = ocrPayload.data.face_session.session_id;
```

Sau khi chụp frame camera thành `Blob` hoặc chọn `File`:

```javascript
const faceForm = new FormData();
faceForm.append("session_id", sessionId);
faceForm.append("capture_source", fromCamera ? "camera" : "upload");
faceForm.append("selfie_image", selfieBlob, "selfie.jpg");

const faceResponse = await fetch(
  "http://localhost:8000/api/face-verification/verify-from-ocr",
  { method: "POST", body: faceForm },
);
const faceResult = await faceResponse.json();
```

Không tự đặt header `Content-Type` khi gửi `FormData`; trình duyệt phải tự
thêm multipart boundary.

## 12. Kiểm tra thủ công trước khi demo đồ án

1. OCR một CCCD rõ nét và xác nhận response có `face_session.can_verify=true`.
2. Mở camera, thử ảnh đúng người: mong đợi `match` hoặc `review` sát ngưỡng.
3. Thử ảnh người khác: mong đợi `not_match`.
4. Cho hai người vào khung: phải lỗi `MULTIPLE_WEBCAM_FACES`.
5. Che camera hoặc quay mặt: phải bị từ chối chất lượng/phát hiện mặt.
6. Upload ảnh thay vì camera: endpoint vẫn chạy và trả
   `capture_source=upload`.
7. Sau `match`, gửi lại cùng session: phải trả HTTP 409.
8. Kiểm tra response luôn có `liveness_checked=false` để báo cáo đúng giới hạn
   của hệ thống.
