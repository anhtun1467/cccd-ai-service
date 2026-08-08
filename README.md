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

Windows/Python 3.11 sử dụng `insightface==1.0.1`. Cài riêng phần Face bằng:

```powershell
python -m pip install --only-binary=:all: -r requirements-face.txt
```

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

## Face Verification Pipeline

```text
Front CCCD Image -> Portrait Detection -> ArcFace 512-D Embedding
Webcam Selfie    -> Exactly One Face -> Quality Check -> ArcFace Embedding
                                        |
                                        v
                          Cosine Similarity
                    MATCH / REVIEW / NOT_MATCH
```

Thresholds:

- `MATCH`: similarity `>= 0.50`
- `REVIEW`: similarity from `0.40` to `< 0.50`
- `NOT_MATCH`: similarity `< 0.40`

API:

```text
POST /api/face-verification/verify
```

Direct Logitech C922 test:

```powershell
python scripts\verify_cccd_with_camera.py `
  --image tests\card_detection\sample_cccd.jpg `
  --camera 0
```

Set `FACE_SAVE_DEBUG=True` only while debugging. The default is `False`
because CCCD and selfie images contain personal data.

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
