# 🇻🇳 CCCD AI Service

AI-powered Vietnamese Citizen Identity Card Recognition and Face Verification System.

---

## Features

- ✅ CCCD Detection
- ✅ Perspective Transform
- ✅ Image Enhancement
- ✅ EasyOCR Integration
- ✅ Text Normalization
- ✅ Regex Parser
- ✅ Data Validation
- ✅ RESTful API
- ✅ Swagger Documentation
- ✅ InsightFace 1:1 Face Verification
- ✅ Face Quality and Multi-face Validation
- ✅ Reuse OCR Session (không upload lại CCCD ở bước Face)
- ✅ Browser Camera + Selfie Upload UI

---

## Tech Stack

- Python 3.11
- FastAPI
- OpenCV
- EasyOCR
- NumPy
- Pydantic
- InsightFace `buffalo_l` / ArcFace
- ONNX Runtime CPU

---

## Project Structure

```
app/
    api/
    core/
    modules/
    services/

storage/
tests/
```

---

## Run

```bash
uvicorn app.main:app --reload
```

Swagger

```
http://localhost:8000/docs
```

Giao diện OCR + camera/upload (mở trực tiếp, không dùng Swagger):

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/face-verification
http://127.0.0.1:8000/camera
```

Địa chỉ gốc `/` và đường dẫn ngắn `/camera` đều tự chuyển tới giao diện có
nút **MỞ CAMERA NGAY**. Trong Swagger cũng có liên kết nổi bật ở phần mô tả.

---

## OCR Pipeline

```
Upload Image
      │
      ▼
Card Detection
      │
      ▼
Perspective Transform
      │
      ▼
Image Enhancement
      │
      ▼
EasyOCR
      │
      ▼
Text Normalizer
      │
      ▼
Regex Parser
      │
      ▼
Validator
      │
      ▼
JSON Response
```

---

## Integrated OCR -> Face Verification Pipeline

```text
POST /ocr/cccd
    -> Card Detection + OCR
    -> cardImage + portrait được giữ ở server
    -> face_session_id (TTL 30 phút, tối đa 5 lần thử)

Camera hoặc selfie upload + face_session_id
    -> POST /api/face-verification/verify-from-ocr
    -> ưu tiên portrait của OCR
    -> fallback cardImage đã làm phẳng
    -> đúng một khuôn mặt + quality check
    -> ArcFace 512-D + cosine similarity
    -> MATCH / REVIEW / NOT_MATCH
```

Thresholds:

- `MATCH`: similarity `>= 0.50`
- `REVIEW`: similarity from `0.40` to `< 0.50`
- `NOT_MATCH`: similarity `< 0.40`

API:

```text
POST   /ocr/cccd
POST   /api/face-verification/verify-from-ocr
GET    /api/face-verification/sessions/{session_id}
DELETE /api/face-verification/sessions/{session_id}
```

Trang camera/upload tích hợp:

```text
http://localhost:8000/face-verification
```

`POST /api/face-verification/verify` vẫn được giữ để tương thích mã cũ,
nhưng endpoint mới không nhận `card_image`; nó chỉ nhận `session_id`,
`selfie_image` và `capture_source` (`camera` hoặc `upload`).

Direct Logitech C922 test:

```powershell
python scripts\verify_cccd_with_camera.py `
  --image tests\card_detection\sample_cccd.jpg `
  --camera 0
```

Set `FACE_SAVE_DEBUG=True` only while debugging. The default is `False`
because CCCD and selfie images contain personal data.

Session settings in `.env` (optional):

```dotenv
FACE_SESSION_TTL_SECONDS=1800
FACE_SESSION_MAX_ATTEMPTS=5
FACE_SESSION_EXPIRED_RETENTION_SECONDS=86400
FACE_SESSION_LEASE_TIMEOUT_SECONDS=300
```

---

## Current Progress

- Card Detection ✅
- OCR Engine ✅
- Parser ✅
- Validator ✅
- REST API ✅
- Face Detection ✅
- Face Verification 1:1 ✅
- Passive image quality checks ✅
- Liveness / anti-spoofing ⏳

---

## Author

Anh Tuan
CMC University
