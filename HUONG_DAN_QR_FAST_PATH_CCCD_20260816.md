# QR Fast Path cho CCCD AI Service

## 1. Mục tiêu

Bản thử nghiệm bổ sung việc đọc QR trên ảnh thẻ đã được cắt và làm phẳng.
QR là nguồn dữ liệu hỗ trợ, không thay thế hoàn toàn OCR:

- QR đọc được: dùng dữ liệu đã qua validator và giảm OCR từng trường.
- QR mờ hoặc không phải QR CCCD: bỏ qua và chạy pipeline OCR cũ.
- QR mâu thuẫn với chữ in hợp lệ: chọn dữ liệu QR nhưng bắt buộc kiểm tra
  thủ công bằng `reviewRequired=true`.

Theo thông tin do Bộ Công an công bố, QR trên thẻ có thể cung cấp số định
danh, họ tên, ngày sinh, giới tính, nơi cư trú, ngày cấp và một số thông tin
bổ sung tùy mẫu thẻ. QR không cung cấp đầy đủ quê quán, quốc tịch và ngày hết
hạn theo hợp đồng JSON hiện tại. Vì vậy hệ thống không gán ngày cấp vào
`dateOfExpiry` và không dùng nơi cư trú thay cho `placeOfOrigin`.

Nguồn tham khảo:

- https://bocongan.gov.vn/chinh-sach-phap-luat/bai-viet/ban-hanh-mau-the-can-cuoc-mau-giay-chung-nhan-can-cuoc-su-dung-tu-0172024-d1-t1415
- https://bocongan.gov.vn/chinh-sach-phap-luat/bai-viet/tao-dieu-kien-thuan-loi-giam-phien-ha-cho-cong-dan-trong-thuc-hien-cac-thu-tuc-hanh-chinh-giao-dich-dan-su-d3-t801

## 2. Các trường QR có thể cấp cho kết quả OCR

| Trường JSON | QR có thể cấp | Xử lý |
| --- | --- | --- |
| `idNumber` | Có | Kiểm tra đúng 12 chữ số |
| `fullName` | Có | Giữ nguyên Unicode và dấu tiếng Việt |
| `dateOfBirth` | Có | Chuẩn hóa `dd/MM/yyyy` |
| `gender` | Có | Chuẩn hóa thành `Nam` hoặc `Nữ` |
| `placeOfResidence` | Có | Phải vượt validator địa chỉ |
| `nationality` | Không | Giữ OCR |
| `placeOfOrigin` | Không | Giữ OCR từng vùng |
| `dateOfExpiry` | Không | Giữ OCR; không lấy ngày cấp QR |

## 3. Cơ chế tối ưu thời gian

QR được giải mã sau khi card detection, orientation và geometry đã chọn ảnh
thẻ cuối cùng. Decoder thử tối đa năm biến thể, có ngân sách mềm mặc định
120 ms.

Nếu một trường QR hợp lệ và full-card OCR không phản đối, OCR từng vùng của
trường đó được bỏ qua. Với QR 7 trường thông thường, tối đa năm trường có thể
không cần field OCR: số CCCD, họ tên, ngày sinh, giới tính và nơi cư trú.

Các trường có xung đột vẫn được OCR riêng để thu thêm bằng chứng.

## 4. Metadata mới

```json
{
  "metadata": {
    "processingStagesMs": {
      "qrFastPath": 42.8
    },
    "dataSources": {
      "idNumber": "CCCD_QR",
      "placeOfOrigin": "FIELD_OCR"
    },
    "qrFastPath": {
      "enabled": true,
      "decoded": true,
      "used": true,
      "attemptCount": 1,
      "format": "CCCD_QR_7_FIELDS",
      "providedFields": [
        "idNumber",
        "fullName",
        "dateOfBirth",
        "gender",
        "placeOfResidence"
      ],
      "skippedFieldOcr": [
        "dateOfBirth",
        "fullName",
        "gender",
        "idNumber",
        "placeOfResidence"
      ],
      "conflicts": []
    }
  }
}
```

Payload QR thô không được đưa vào response hoặc log benchmark.

## 5. Cấu hình `.env`

```dotenv
QR_FAST_PATH_ENABLED=true
QR_DECODE_BUDGET_MS=120
QR_SKIP_CONFIRMED_FIELD_OCR=true
```

Đặt `QR_FAST_PATH_ENABLED=false` để quay về pipeline chỉ OCR mà không cần sửa
mã nguồn.

## 6. Chạy kiểm thử trên Windows PowerShell

```powershell
pytest .\tests\qr -v
pytest -q
```

Benchmark một ảnh mà không nạp model EasyOCR:

```powershell
python .\scripts\benchmark_qr_fast_path.py `
  ".\storage\uploads\mau_cccd.jpg" `
  --repeat 3
```

Benchmark cả thư mục:

```powershell
python .\scripts\benchmark_qr_fast_path.py `
  ".\storage\uploads" `
  --repeat 3
```

Kết quả benchmark chỉ báo QR đọc được hay không, thời gian và tên các trường;
không in giá trị CCCD, họ tên hoặc địa chỉ.

## 7. Giới hạn của bản thử nghiệm

- Hai ảnh mẫu chụp xa/mờ hiện không giải mã được QR bằng OpenCV, nên vẫn chạy
  OCR cũ và chỉ phát sinh thêm khoảng 59–69 ms ở bước QR.
- QR rõ giả lập giải mã có trung vị khoảng 43 ms trong môi trường kiểm thử.
- Máy quét QR USB chuyên dụng thường ổn định hơn camera, nhưng luồng nhận chuỗi
  từ thiết bị USB chưa nằm trong bản thử nghiệm này.
- QR giúp giảm lỗi ở những trường có trong QR nhưng không sửa trực tiếp thuật
  toán card detection hoặc ảnh quá mờ.
