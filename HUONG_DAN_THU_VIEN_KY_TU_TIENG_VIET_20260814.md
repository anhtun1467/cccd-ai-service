# Thư viện ký tự tiếng Việt cho OCR CCCD

## 1. Mục tiêu

Tầng mới đối chiếu hình dáng ký tự trong từng `OCRTextBox` với thư viện mẫu
ảnh trước khi `FieldOCRService` ghép dòng, làm sạch và chuyển kết quả cho
`Result Fuser`.

Luồng xử lý:

```text
Field crop
→ EasyOCR + allowlist theo field
→ cắt từng OCR box/ký tự
→ so khớp atlas glyph
→ chỉ sửa khi đủ ngưỡng an toàn
→ Line Merger
→ Text Normalizer
→ Parser / Result Fuser / Validator
```

Đây là tầng kiểm chứng ảnh, không phải từ điển đoán họ tên/địa chỉ. Khi ảnh
không đủ nét hoặc hai dấu quá giống nhau, hệ thống giữ nguyên EasyOCR và đưa
ứng viên vào `reviewCandidates`.

## 2. Nội dung thư viện

- 89 chữ thường tiếng Việt.
- 89 chữ hoa tiếng Việt.
- Đầy đủ các họ nguyên âm:
  - `a, ă, â` cùng sáu thanh.
  - `e, ê` cùng sáu thanh.
  - `i`, `y` cùng sáu thanh.
  - `o, ô, ơ` cùng sáu thanh.
  - `u, ư` cùng sáu thanh.
- `d/đ`, toàn bộ chữ cái ASCII để đọc nhãn song ngữ.
- Chữ số `0-9`.
- Dấu câu dùng trong CCCD và địa chỉ: `/ . , : ; - ' ( )` cùng một số dấu
  mở rộng.
- Mọi chuỗi được chuẩn hóa Unicode NFC; ví dụ chuỗi tổ hợp `a + breve + sắc`
  được đưa về một ký tự `ắ`.

Atlas bàn giao có 216 ký tự duy nhất và 864 mẫu ảnh 64 × 64 từ bốn font
regular/bold, sans/serif. `manifest.json` chứa SHA-256; atlas sai hoặc bị thiếu
sẽ tự bị vô hiệu hóa, không làm endpoint OCR lỗi.

## 3. Bảng ký tự theo field

| Field | Ký tự được phép | Cách đối chiếu |
| --- | --- | --- |
| `idNumber` | `0-9` | So toàn bộ chữ số; hỗ trợ `O→0`, `I/l→1`, `S→5`, `B→8` khi đủ điểm |
| `dateOfBirth`, `dateOfExpiry` | `0-9 / . -` | So số, sau đó vẫn kiểm tra ngày hợp lệ |
| `fullName` | Chữ Việt hoa/thường, khoảng trắng, `-`, `'` | So trong cùng họ nguyên âm và `D/Đ` |
| `gender`, `nationality` | Chữ Việt + ASCII | Matcher bổ trợ; normalizer hiện tại vẫn là lớp quyết định cuối |
| `placeOfOrigin`, `placeOfResidence` | Chữ Việt, số và dấu câu địa chỉ | So glyph trước, sau đó fuser/validator loại nhiễu chéo field |

## 4. Điều kiện tự sửa

Một ký tự chỉ được thay khi đồng thời thỏa mãn:

1. OCR box cao tối thiểu 16 px.
2. Tách được số đoạn ảnh phù hợp với số ký tự OCR.
3. Độ tin cậy phân đoạn từ `0.70` trở lên.
4. Điểm mẫu chữ tối thiểu `0.78`; chữ số tối thiểu `0.76`.
5. Ứng viên tốt nhất cách ứng viên thứ hai ít nhất `0.020` đối với chữ và
   `0.075` đối với số.
6. Ứng viên tốt nhất phải cải thiện đủ so với ký tự EasyOCR ban đầu.
7. Các từ nhãn tiếng Anh như `Full`, `Name`, `Date`, `Birth`, `Place`,
   `Residence` được khóa, không tự thêm dấu.
8. Box phân đoạn yếu chỉ được sửa tối đa 45% ký tự đã so; box rõ tối đa 75%.

Khi thiếu dấu nhưng thân chữ giống nhau rất mạnh, hệ thống có ngưỡng riêng để
khôi phục các ca như `BUI THI DUYEN → BÙI THỊ DUYÊN`. Ca mơ hồ như sắc/hỏi
trên ảnh quá nhòe được giữ nguyên thay vì đoán.

## 5. Dữ liệu debug mới

Mỗi candidate trong `fieldResults.<field>.ocrCandidates` có:

```json
{
  "glyphRefinedText": ["BÙI THỊ DUYÊN"],
  "glyphMatch": {
    "available": true,
    "atlasVersion": "2026.08.14",
    "comparedCharacters": 7,
    "confidentCharacters": 7,
    "averageBestScore": 0.94,
    "coverage": 1.0,
    "applied": true,
    "corrections": [
      {
        "from": "U",
        "to": "Ù",
        "score": 0.95,
        "originalScore": 0.68,
        "margin": 0.03
      }
    ],
    "reviewCandidates": []
  }
}
```

`reviewCandidates.decision` có thể là:

- `APPLIED`: đã sửa.
- `AMBIGUOUS_TOP_CANDIDATES`: hai ký tự quá sát điểm.
- `LOW_TEMPLATE_SCORE`: ảnh không giống đủ mạnh.
- `LOW_IMPROVEMENT_OVER_OCR`: mẫu mới không tốt hơn EasyOCR đủ nhiều.
- `LOW_SEGMENTATION_CONFIDENCE`: không tin cậy vị trí ký tự.
- `REPORT_ONLY`: đã so sánh nhưng cấu hình không cho tự sửa.
- `CHANGE_LIMIT`: đề xuất vượt giới hạn số ký tự được đổi trong một box.

## 6. Cấu hình `.env`

Mặc định matcher và tự sửa đều bật:

```dotenv
OCR_GLYPH_MATCHER_ENABLED=true
OCR_GLYPH_AUTO_CORRECT=true
```

Muốn thu thập điểm thực tế mà chưa thay kết quả OCR:

```dotenv
OCR_GLYPH_MATCHER_ENABLED=true
OCR_GLYPH_AUTO_CORRECT=false
```

Muốn quay lại hoàn toàn luồng cũ:

```dotenv
OCR_GLYPH_MATCHER_ENABLED=false
```

Sau khi đổi `.env`, khởi động lại Uvicorn.

## 7. Kiểm thử

Chạy riêng 14 test mới:

```powershell
python -m pytest `
  tests\ocr\test_vietnamese_charset.py `
  tests\ocr\test_vietnamese_glyph_matcher.py `
  tests\ocr\test_field_glyph_integration.py `
  -v
```

Chạy hồi quy toàn dự án:

```powershell
python -m pytest
```

Kiểm tra một crop/box thật:

```powershell
python scripts\test_vietnamese_glyph_matcher.py `
  --image "storage\debug\<request-id>\fields\fullName.jpg" `
  --text "BUI THI DUYEN" `
  --field fullName `
  --box 0 0 760 90
```

Nếu bỏ `--box`, script dùng toàn bộ ảnh. Với crop rộng chứa hai field, nên
truyền đúng bounding box của dòng giá trị để kết quả phân đoạn có ý nghĩa.

## 8. Tạo lại atlas trên Windows

Atlas kèm theo đã dùng font Unicode tự do. Trên máy chạy thật có thể tạo lại
bằng Arial, Tahoma và Segoe UI sẵn có trong Windows:

```powershell
python scripts\build_vietnamese_glyph_library.py
```

Hoặc chỉ định font:

```powershell
python scripts\build_vietnamese_glyph_library.py `
  --font "C:\Windows\Fonts\arial.ttf" `
  --font "C:\Windows\Fonts\arialbd.ttf" `
  --font "C:\Windows\Fonts\tahoma.ttf" `
  --font "C:\Windows\Fonts\tahomabd.ttf"
```

Script kiểm tra từng font trên toàn bộ bảng ký tự. Font thiếu một glyph tiếng
Việt sẽ bị bỏ, và checksum mới được ghi vào manifest tự động.

## 9. Giới hạn cần đánh giá bằng ảnh thật

- Template matching không thay thế mô hình OCR đã huấn luyện.
- Dấu sắc/hỏi/ngã ở ảnh nhỏ hoặc lóa có thể không đủ pixel để phân biệt.
- Box có hai dòng hoặc dính nhãn cần được EasyOCR tách đúng trước.
- Nên chạy Ground Truth riêng cho tên và địa chỉ, so sánh ba chế độ:
  `matcher off`, `report-only`, `auto-correct` bằng Field Accuracy, CER và số
  correction sai.
- Chỉ giảm ngưỡng sau khi có thống kê trên tập validation; không hạ ngưỡng
  dựa trên một ảnh đơn lẻ.
