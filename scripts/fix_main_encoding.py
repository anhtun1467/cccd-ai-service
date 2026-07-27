from pathlib import Path


path = Path("app/main.py")
raw = path.read_bytes()

encodings = (
    "utf-8-sig",
    "utf-8",
    "cp1258",
    "cp1252",
    "latin-1",
)

content = None
detected_encoding = None

for encoding in encodings:
    try:
        content = raw.decode(encoding)
        detected_encoding = encoding
        break
    except UnicodeDecodeError:
        continue

if content is None:
    raise RuntimeError("Không thể xác định encoding của app/main.py")

# Xóa Markdown fence nếu từng dán code từ ChatGPT
lines = [
    line
    for line in content.splitlines()
    if line.strip() not in {"```", "```python", "```py"}
]

content = "\n".join(lines).rstrip() + "\n"

path.write_text(
    content,
    encoding="utf-8",
    newline="\n",
)

print(
    f"Đã chuyển app/main.py: "
    f"{detected_encoding} -> utf-8"
)
