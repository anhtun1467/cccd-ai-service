# Thư viện glyph tiếng Việt cho OCR CCCD

Thư mục này chứa atlas mẫu ảnh của toàn bộ 89 chữ thường, 89 chữ hoa,
10 chữ số và các dấu câu dùng trên mặt trước CCCD.

- `vietnamese_glyph_templates.npz`: mẫu ảnh nhị phân 64 x 64 từ nhiều font.
- `manifest.json`: phiên bản, danh sách ký tự, font nguồn và SHA-256.

Matcher chỉ dùng atlas để xác nhận hoặc sửa ký tự khi ảnh crop đủ nét và
điểm tốt nhất vượt các ngưỡng an toàn. Nếu atlas thiếu/hỏng, OCR vẫn chạy và
giữ nguyên kết quả EasyOCR.

Tạo lại atlas trên Windows để bổ sung Arial, Tahoma và Segoe UI:

```powershell
python scripts\build_vietnamese_glyph_library.py
```

Có thể truyền thêm font gần với dữ liệu thực tế:

```powershell
python scripts\build_vietnamese_glyph_library.py `
  --font "C:\Windows\Fonts\arial.ttf" `
  --font "C:\Windows\Fonts\arialbd.ttf"
```
