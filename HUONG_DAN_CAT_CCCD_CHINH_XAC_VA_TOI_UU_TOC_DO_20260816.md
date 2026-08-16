# Bản sửa cắt CCCD chính xác và tối ưu thời gian — 16/08/2026

## Kết quả chính

Bản này xử lý đồng thời hai lỗi: cắt nhầm một hình chữ nhật bên trong thẻ
(QR, ảnh chân dung hoặc cụm chữ) và chạy EasyOCR lặp quá nhiều lần.

- Ảnh chụp xa vẫn dùng bốn cạnh Hough khi threshold sáng dính thẻ vào tay,
  màn hình hoặc mặt bàn.
- Ảnh đã có contour trọn thẻ bỏ qua Hough hoàn toàn.
- Một ứng viên Hough chỉ được giữ khi đủ lớn so với contour chính, chồng phủ
  gần trọn vùng chính, tâm không lệch xa và chi tiết trải đủ lưới 4 x 3.
- Chuẩn hóa dấu pháp tuyến của `cv2.HoughLines`, sửa trường hợp hai cạnh thật
  nằm gần góc 0/180 độ nhưng trước đây bị hiểu là cách nhau vài pixel.
- Chỉ một ứng viên hình học đã kiểm chứng được phép chạy OCR; không còn ba
  lần OCR cho QR/vùng chữ hoặc ba phép nắn gần giống nhau.
- Xoay thẻ đúng chiều trước khi so sánh ứng viên hình học. Ảnh úp ngược không
  còn làm mọi ứng viên bị chấm điểm thấp rồi OCR lại hàng loạt.

## Cắt từng trường theo nhãn thật

Ngoài độ lệch chung theo số CCCD, pipeline nay tìm riêng các nhãn:

1. Số / No.
2. Họ và tên / Full name.
3. Ngày sinh / Date of birth.
4. Giới tính / Sex và Quốc tịch / Nationality.
5. Quê quán / Place of origin.
6. Nơi thường trú / Place of residence.

Mép nhãn hàng dưới trở thành ranh giới chung của hai hàng. Vì vậy vùng họ
tên không chồng vào ngày sinh, ngày sinh không chồng vào giới tính, và quê
quán kết thúc đúng nơi vùng thường trú bắt đầu. Khi không nhận ra nhãn, hệ
thống tự quay về template có giới hạn an toàn. Vùng chân dung cũng được nới
đến `x=310, y=545` trên ảnh chuẩn 1000 x 630 để không cắt mất vai/mép mặt.

Thông tin debug mới nằm tại:

- `metadata.orientation.fieldCropLayout`
- `metadata.orientation.addressCropLayout`
- `metadata.fieldDebug.fieldLayout`

## Tối ưu EasyOCR

- Ảnh field đã phóng trong cropper không bị EasyOCR phóng tiếp 1,8 lần.
- Ảnh processed dùng `mag_ratio=1.0`; ảnh raw dùng `1.6`.
- `canvas_size` field giảm từ 3200 xuống 2048 và recognition batch tăng từ
  1 lên 4.
- Ảnh đủ sáng thử raw trước; ảnh tối dưới ngưỡng vẫn thử CLAHE/processed
  trước để giữ khả năng đọc ảnh thiếu sáng.
- Nếu một lần OCR field khớp dữ liệu OCR toàn thẻ, họ tên/địa chỉ/ngày hết
  hạn có thể dừng sớm bằng đồng thuận hai nguồn.

## Số liệu kiểm tra tại workspace

- Mẫu chụp xa của người dùng: detector khoảng 1,48 giây trước sửa, còn
  khoảng 0,32–0,37 giây; crop cuối 604 x 381 pixel, không còn tay/nền.
- Ảnh đủ khung trong bộ debug: Hough từ khoảng 100–600 ms còn khoảng
  4–10 ms vì được bỏ qua.
- Ba mẫu debug có contour dính nền/khung đều tạo đúng một crop Hough toàn
  thẻ; các hình chữ nhật nội bộ bị loại.
- Phát lại kích thước 312 lần OCR field trong JSON debug cho thấy lượng pixel
  đầu vào hiệu dụng giảm từ khoảng 267,9 MP xuống 103,5 MP, tương đương
  61,4%. Đây là chỉ số tải tính toán, không phải cam kết thời gian tuyệt đối.
- Bộ test: 188 passed, 1 skipped.

Thời gian cuối phụ thuộc CPU/model EasyOCR. Sau khi chạy thật, xem:

- `metadata.processingStagesMs`
- `metadata.fullCardOcrAttemptCount`
- `metadata.fieldOcrAttemptCount`

để biết chính xác thời gian nằm ở detector, xoay/hình học hay OCR field.

## Chạy kiểm thử và benchmark

```powershell
pytest -q
python scripts/benchmark_crop_precision_speed.py `
  "duong-dan-anh-1.jpg" "thu-muc-debug" --repeat 3
```

Benchmark chỉ chạy phần phát hiện/cắt thẻ, không nạp model EasyOCR nên phù
hợp để so sánh nhanh trên máy triển khai.

## Tệp runtime đã thay đổi

- `app/modules/card_detection/contour_detector.py`
- `app/modules/card_detection/detector.py`
- `app/modules/ocr/result_fuser.py`
- `app/modules/ocr/field_cropper.py`
- `app/modules/ocr/field_ocr_service.py`
- `app/modules/ocr/easyocr_engine.py`
- `app/services/ocr_pipeline.py`

Gói bàn giao không chứa CCCD thật, JSON OCR thật, thư mục `storage/debug`,
model hoặc virtual environment.
