from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

file_path = (
    ROOT_DIR
    / "app"
    / "modules"
    / "ocr"
    / "result_fuser.py"
)

content = file_path.read_text(
    encoding="utf-8-sig"
)

start_marker = "def select_date("
end_marker = "\ndef select_nationality("

start_index = content.find(start_marker)

end_index = content.find(
    end_marker,
    start_index,
)

if start_index == -1:
    raise RuntimeError(
        "Không tìm thấy hàm select_date trong result_fuser.py."
    )

if end_index == -1:
    raise RuntimeError(
        "Không tìm thấy hàm select_nationality "
        "sau hàm select_date."
    )

new_function = r'''def select_date(
    field_value: str | None,
    full_card_value: str | None,
    raw_text: list[str] | None,
) -> tuple[str | None, str]:
    """
    Chọn và chuẩn hóa ngày sinh từ nhiều nguồn OCR.

    Thứ tự ưu tiên:
        1. OCR theo vùng field.
        2. OCR toàn bộ CCCD.
        3. Phục hồi từ từng dòng raw text.
    """

    field_date = normalize_date(
        field_value
    )

    if field_date:
        return field_date, "FIELD_OCR"

    full_card_date = normalize_date(
        full_card_value
    )

    if full_card_date:
        return (
            full_card_date,
            "FULL_CARD_OCR",
        )

    for line in raw_text or []:
        if not line:
            continue

        line_text = str(line)

        # Ví dụ:
        # Ngay sinh / Date of birth: 24/0311995
        #
        # Phần được lấy:
        # 24/0311995
        date_candidates = re.findall(
            r"(?<!\d)"
            r"\d{1,2}"
            r"\s*[./\\-]\s*"
            r"\d{2,7}"
            r"(?!\d)",
            line_text,
        )

        # Hỗ trợ ngày mất toàn bộ dấu phân cách:
        # 24031995
        #
        # Hoặc OCR bị thừa một chữ số:
        # 240311995
        compact_candidates = re.findall(
            r"(?<!\d)"
            r"\d{8,9}"
            r"(?!\d)",
            line_text,
        )

        date_candidates.extend(
            compact_candidates
        )

        for candidate in date_candidates:
            recovered_date = normalize_date(
                candidate
            )

            if recovered_date:
                return (
                    recovered_date,
                    "RAW_TEXT_RECOVERY",
                )

    return None, "NOT_FOUND"

'''

updated_content = (
    content[:start_index]
    + new_function
    + content[end_index:]
)

file_path.write_text(
    updated_content,
    encoding="utf-8",
)

print("=" * 64)
print("ĐÃ CẬP NHẬT SELECT_DATE THÀNH CÔNG")
print("=" * 64)
print(file_path)
