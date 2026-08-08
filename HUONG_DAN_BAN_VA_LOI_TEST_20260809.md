# Bản vá lỗi OCR từ bộ test ngày 09/08/2026

## Cài bản vá

Giải nén ZIP trực tiếp vào thư mục gốc của dự án và cho phép ghi đè:

```powershell
Expand-Archive -LiteralPath .\cccd_ocr_fix_test_20260809.zip -DestinationPath . -Force
```

Nếu đang dùng môi trường ảo, hãy kích hoạt môi trường đó trước khi test.

## Chạy kiểm thử

```powershell
python -m pytest
```

File `pytest.ini` giới hạn pytest trong thư mục `tests`, tránh thu thập lại
các test trùng tên trong thư mục backup cũ.

Khởi động lại API sau khi test đạt:

```powershell
python -m uvicorn app.main:app --reload
```

## Những lỗi đã sửa

- Bộ hợp nhất kết quả dùng toàn bộ `ocrCandidates` của từng trường, thay vì
  chỉ nhìn một giá trị crop đã được chọn sớm.
- Họ tên ưu tiên dòng có nhãn và chỉ dùng đồng thuận crop để sửa sai lệch
  rất gần; mẩu nhãn ngày sinh không còn bị nối vào tên.
- Ngày sinh/ngày hết hạn được đối chiếu cấu trúc số CCCD và các mốc tuổi,
  nhưng chỉ phục hồi khi OCR thực sự có bằng chứng số tương ứng.
- Địa chỉ được khôi phục dấu, sửa thứ tự số nhà/tên đường, loại phần ngày
  hết hạn bị trộn vào địa chỉ và ghép lại dòng bị xen giữa hai cột.
- Bộ ghép dòng dùng tâm/chiều cao trung vị để một box cao bất thường không
  nối hai dòng địa chỉ thành một dòng sai.
- Bổ sung crop hẹp cho hạn sử dụng và tăng số biến thể OCR cho các trường
  tên, địa chỉ và ngày hết hạn.

## Kết quả đối chiếu bộ dữ liệu đã gửi

- 9/10 ảnh cho dữ liệu đầy đủ và qua validator sau khi hợp nhất lại JSON.
- Ảnh `b5d3...` vẫn chủ động không hợp lệ: mép dưới thẻ đã bị cắt, chỉ thấy
  `Thôn 4B` và không có ngày hết hạn. Hệ thống giữ `dateOfExpiry = null`
  thay vì tự đoán dữ liệu không xuất hiện trên ảnh.
- 46 kiểm thử thuần Python liên quan đến hợp nhất, validator và ghép dòng
  đã đạt; 24 JSON hồi quy cũ tăng từ 9 lên 19 kết quả hợp lệ, không làm một
  kết quả hợp lệ cũ nào trở thành không hợp lệ.

Để ảnh `b5d3...` có kết quả đầy đủ, cần chụp lại toàn bộ mặt trước CCCD,
đặc biệt là cạnh dưới và dòng ngày hết hạn.
