# Cập nhật OCR cho ảnh thiếu sáng và mờ nhẹ

## Kết quả của bản cập nhật

Pipeline không còn loại ảnh chỉ vì ảnh hơi tối hoặc hơi mờ ở bước upload.
Ảnh còn nhìn thấy nội dung sẽ đi qua CardDetector, được nắn hình học, rồi
pipeline tạo các biến thể OCR phù hợp với đúng lỗi ảnh:

- `low_light`: nâng sáng thích ứng trên kênh độ sáng, giữ màu của thẻ;
- `mild_deblur`: tăng nét có ngưỡng, hạn chế viền kép và nhiễu nền;
- `low_light_deblur`: kết hợp hai bước khi ảnh vừa tối vừa mờ;
- `balanced`: chỉ thử khi OCR ban đầu yếu nhưng chỉ số ảnh chưa vượt ngưỡng.

Ảnh warped gốc vẫn là ảnh chính. Mỗi biến thể phải thắng điểm OCR của ảnh
gốc ít nhất `0.45` mới được chọn. Vì vậy ảnh đã rõ không bị tăng nét hoặc
tăng tương phản quá tay.

## Các ngưỡng đang dùng

- Upload chỉ chặn khung gần như đen: brightness dưới `15`.
- Vùng thẻ cảnh báo thiếu sáng: brightness dưới `120`.
- Tạo biến thể toàn thẻ khi brightness dưới `128`, p90 dưới `170`, hoặc
  Laplacian dưới `110`.
- Tạo biến thể từng field khi brightness dưới `138`, p90 dưới `180`, hoặc
  Laplacian dưới `105`.
- Ảnh hơi mờ/thiếu sáng vẫn được trả kết quả khi validator xác nhận tối
  thiểu hai trường lõi và OCR có đủ dòng chữ có nghĩa.

Các ngưỡng tạo ứng viên không phải ngưỡng từ chối. Quyết định cuối dựa vào
OCR, validator và điểm của từng phương án ảnh.

## Ảnh gốc và Face Verification

Các biến thể tăng sáng/khử mờ chỉ dùng cho OCR. `cardImage`, vùng chân dung
và Face Verification tiếp tục dùng ảnh warped gốc. Tọa độ OCR toàn thẻ
không lệch vì mọi biến thể đều giữ nguyên chiều rộng, chiều cao.

## Tệp debug cần xem khi kiểm tra

- `detector_03_warped.jpg`: ảnh gốc sau nắn thẻ;
- `detector_07_quality_*.jpg`: các phương án thích ứng đã thử;
- `detector_07_quality_selected.jpg`: phương án được chọn;
- `fields/*_low_light*.png`: crop field thiếu sáng;
- `fields/*_deblur*.png`: crop field mờ nhẹ;
- `metadata.orientation.qualityEnhancement`: điểm và lý do chọn;
- `metadata.imageQuality.adaptiveProfile`: chỉ số sáng/mờ đo được.

## Chạy kiểm thử

```bash
pytest -q \
  tests/card_detection/test_adaptive_quality_enhancer.py \
  tests/ocr/test_low_light_field_variants.py \
  tests/test_reject_image.py
```

Benchmark suy giảm ảnh có thể chạy bằng:

```bash
python scripts/benchmark_adaptive_quality.py \
  --image-dir datasets/source/images \
  --output storage/debug/adaptive_quality_benchmark.json
```

Benchmark trên chỉ đo độ sáng, độ nét và việc giữ kích thước. Không được
dùng nó để tuyên bố tỷ lệ đọc đúng OCR. Muốn đo CER, WER hoặc Exact Match
cần một manifest ground truth đã kiểm tra thủ công.

## Giới hạn thực tế

Bộ tăng cường giúp với thiếu sáng và mất nét nhẹ. Ảnh rung mạnh, chữ đã hòa
vào nền hoặc vùng thẻ chỉ còn vài pixel không thể khôi phục thông tin không
còn tồn tại; pipeline sẽ trả `OCR_UNREADABLE_IMAGE` thay vì dựng dữ liệu giả.
