from pathlib import Path


file_path = Path(
    "app/modules/ocr/result_fuser.py"
)

content = file_path.read_text(
    encoding="utf-8-sig"
)

start_marker = "def normalize_date("
end_marker = "`ndef is_valid_full_name("

start_index = content.find(start_marker)
end_index = content.find(
    end_marker,
    start_index,
)

if start_index == -1:
    raise RuntimeError(
        "Không tìm thấy hàm normalize_date."
    )

if end_index == -1:
    raise RuntimeError(
        "Không tìm thấy hàm is_valid_full_name."
    )

new_function = r'''def normalize_date(value: str | None) -> str | None:
    """
    Chuẩn hóa ngày tháng từ kết quả OCR.

    Hỗ trợ:
        24/03/1995
        24-03-1995
        24.03.1995
        24031995
        24/031995
        24/0311995

    Hàm chỉ phân tích các cụm số, tránh đổi chữ trong
    nhãn OCR như "Ngay sinh / Date of birth" thành số.
    """

    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    def build_date(
        day: str,
        month: str,
        year: str,
    ) -> str | None:
        try:
            day_number = int(day)
            month_number = int(month)
            year_number = int(year)
        except ValueError:
            return None

        candidate = (
            f"{day_number:02d}/"
            f"{month_number:02d}/"
            f"{year_number:04d}"
        )

        try:
            parsed_date = datetime.strptime(
                candidate,
                "%d/%m/%Y",
            )
        except ValueError:
            return None

        if not 1900 <= parsed_date.year <= 2100:
            return None

        return candidate

    # 1. Ngày có đủ dấu phân cách: 24/03/1995
    standard_patterns = (
        r"(?<!\d)"
        r"(\d{1,2})"
        r"\s*[./\\-]\s*"
        r"(\d{1,2})"
        r"\s*[./\\-]\s*"
        r"(\d{4})"
        r"(?!\d)",

        # Ngày tháng có thể dính: 2403/1995
        r"(?<!\d)"
        r"(\d{2})"
        r"(\d{2})"
        r"\s*[./\\-]\s*"
        r"(\d{4})"
        r"(?!\d)",

        # Tháng năm có thể dính: 24/031995
        r"(?<!\d)"
        r"(\d{2})"
        r"\s*[./\\-]\s*"
        r"(\d{2})"
        r"(\d{4})"
        r"(?!\d)",
    )

    for pattern in standard_patterns:
        match = re.search(
            pattern,
            text,
        )

        if not match:
            continue

        normalized_date = build_date(
            *match.groups()
        )

        if normalized_date:
            return normalized_date

    # 2. Thu thập riêng các cụm số hoặc dấu ngày tháng.
    numeric_chunks = re.findall(
        r"(?<!\d)"
        r"\d[\d\s./\\-]{6,12}\d"
        r"(?!\d)",
        text,
    )

    # Trường hợp đầu vào chỉ là một giá trị ngày.
    if not numeric_chunks:
        numeric_chunks = [text]

    for chunk in numeric_chunks:
        digits = "".join(
            re.findall(
                r"\d",
                chunk,
            )
        )

        # Chuỗi ngày đủ 8 số: 24031995.
        if len(digits) == 8:
            normalized_date = build_date(
                digits[0:2],
                digits[2:4],
                digits[4:8],
            )

            if normalized_date:
                return normalized_date

        # OCR thừa một chữ số: 240311995.
        if len(digits) == 9:
            candidates: list[str] = []

            # Ưu tiên xóa một ký tự trong cặp lặp.
            for index in range(
                len(digits) - 1
            ):
                if digits[index] == digits[index + 1]:
                    candidates.append(
                        digits[:index]
                        + digits[index + 1:]
                    )

            # Phương án dự phòng: thử xóa từng chữ số.
            for index in range(len(digits)):
                candidate = (
                    digits[:index]
                    + digits[index + 1:]
                )

                if candidate not in candidates:
                    candidates.append(candidate)

            for candidate in candidates:
                normalized_date = build_date(
                    candidate[0:2],
                    candidate[2:4],
                    candidate[4:8],
                )

                if normalized_date:
                    return normalized_date

    return None

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

print(
    "Đã cập nhật normalize_date phiên bản 3."
)
