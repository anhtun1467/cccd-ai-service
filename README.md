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

---

## Tech Stack

- Python 3.11
- FastAPI
- OpenCV
- EasyOCR
- NumPy
- Pydantic

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

## Current Progress

- Card Detection ✅
- OCR Engine ✅
- Parser ✅
- Validator ✅
- REST API ✅

---

## Author

Anh Tuan
CMC University