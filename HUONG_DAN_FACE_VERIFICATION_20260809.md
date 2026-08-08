# Hướng dẫn Face Verification

## 1. Cài thư viện

```powershell
cd "C:\Users\Ahn Tuaann\cccd-ai-service"
.\venv\Scripts\Activate.ps1
python -m pip install --only-binary=:all: -r requirements-face.txt
```

Model mặc định: `InsightFace buffalo_l`, provider CPU.

Trên Windows/Python 3.11, dùng `insightface==1.0.1`. Không hạ xuống
`insightface==0.7.3`: bản `0.7.3` chỉ phát hành source distribution nên pip sẽ
cố biên dịch C/Cython và yêu cầu Microsoft Visual C++ Build Tools.

Kiểm tra môi trường sau khi cài:

```powershell
python -m pip check
python -c "import insightface, onnxruntime as ort; print('InsightFace:', insightface.__version__); print('ONNX Runtime:', ort.__version__); print('Providers:', ort.get_available_providers())"
```

Kết quả CPU tối thiểu phải có `CPUExecutionProvider`.

## 2. Chạy kiểm thử

```powershell
python -m pytest tests\face_verification -v
python -m pytest
```

Kiểm thử webcam thật được bỏ qua mặc định để không làm hỏng bộ test trên máy
không có camera. Khi cần kiểm tra Logitech C922:

```powershell
$env:RUN_CAMERA_TEST="1"
python -m pytest tests\face_verification\test_camera_source.py -v
Remove-Item Env:RUN_CAMERA_TEST
```

## 3. Chạy API

```powershell
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Mở `http://127.0.0.1:8000/docs`, chọn:

```text
POST /api/face-verification/verify
```

- `card_image`: ảnh mặt trước CCCD.
- `webcam_image`: ảnh selfie chỉ có một người, nhìn thẳng, rõ nét và đủ sáng.

Swagger chỉ gửi file ảnh; trình duyệt không cho backend FastAPI tự điều khiển
webcam của máy khách. Frontend React phải dùng `getUserMedia()` để chụp frame
rồi gửi frame đó vào trường `webcam_image`.

## 4. Chạy trực tiếp với Logitech C922

```powershell
python scripts\verify_cccd_with_camera.py `
  --image tests\card_detection\sample_cccd.jpg `
  --camera 0 `
  --match-threshold 0.50 `
  --review-threshold 0.40
```

- Nhấn `S` để chụp.
- Nhấn `Q` hoặc `ESC` để hủy.
- Script lấy một cụm frame và chọn frame nét nhất trước khi so khớp.

## 5. Phân loại kết quả

| Similarity | Kết quả |
| --- | --- |
| `>= 0.50` | `match` nếu chất lượng ảnh đạt |
| `0.40 - < 0.50` | `review` |
| `< 0.40` | `not_match` |

Nếu similarity đạt `MATCH` nhưng ảnh hơi mờ, lệch góc hoặc sai sáng, kết quả
được hạ xuống `review` thay vì tự động chấp nhận.

## 6. Các lỗi đầu vào quan trọng

- `CCCD_FACE_NOT_FOUND`: không tìm thấy chân dung trên CCCD.
- `WEBCAM_FACE_NOT_FOUND`: không tìm thấy khuôn mặt webcam.
- `MULTIPLE_WEBCAM_FACES`: ảnh webcam có từ hai khuôn mặt trở lên.
- `WEBCAM_FACE_TOO_BLURRY`: ảnh quá mờ.
- `WEBCAM_FACE_TOO_DARK` / `WEBCAM_FACE_TOO_BRIGHT`: ánh sáng không đạt.
- `WEBCAM_FACE_YAW_INVALID`, `PITCH_INVALID`, `ROLL_INVALID`: góc mặt quá lớn.
- `FACE_MODEL_UNAVAILABLE`: model InsightFace hoặc ONNX Runtime chưa sẵn sàng.

## 7. Dữ liệu debug và liveness

Mặc định API không lưu ảnh CCCD/selfie. Khi cần debug tạm thời, thêm vào `.env`:

```dotenv
FACE_SAVE_DEBUG=True
```

Sau khi debug xong nên đổi lại `False` và xóa ảnh nhạy cảm trong
`storage/debug/face_verification_api`.

Endpoint hiện so khớp từ một ảnh tĩnh nên trường `liveness_checked` luôn là
`false`. Kiểm tra chất lượng ảnh không thay thế cho anti-spoofing/liveness.
