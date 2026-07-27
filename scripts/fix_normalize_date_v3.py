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
        "Không t́m th?y hàm normalize_date."
    )

if end_index == -1:
    raise RuntimeError(
        "Không t́m th?y hàm is_valid_full_name."
    )

new_function = r'''def normalize_date(value: str | None) -> str | None:
    """
    Chu?n hóa ngày tháng t? k?t qu? OCR.

    H? tr?:
        24/03/1995
        24-03-1995
        24.03.1995
        24031995
        24/031995
        24/0311995

    Hàm ch? phân tích các c?m s?, tránh d?i ch? trong
    nhăn OCR nhu "Ngay sinh / Date of birth" thành s?.
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

    # 1. Ngày có d? d?u phân cách: 24/03/1995
    standard_patterns = (
        r"(?<!\d)"
        r"(\d{1,2})"
        r"\s*[./\\-]\s*"
        r"(\d{1,2})"
        r"\s*[./\\-]\s*"
        r"(\d{4})"
        r"(?!\d)",

        # Ngày tháng có th? dính: 2403/1995
        r"(?<!\d)"
        r"(\d{2})"
        r"(\d{2})"
        r"\s*[./\\-]\s*"
        r"(\d{4})"
        r"(?!\d)",

        # Tháng nam có th? dính: 24/031995
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

    # 2. Thu th?p riêng các c?m s? ho?c d?u ngày tháng.
    numeric_chunks = re.findall(
        r"(?<!\d)"
        r"\d[\d\s./\\-]{6,12}\d"
        r"(?!\d)",
        text,
    )

    # Tru?ng h?p d?u vào ch? là m?t giá tr? ngày.
    if not numeric_chunks:
        numeric_chunks = [text]

    for chunk in numeric_chunks:
        digits = "".join(
            re.findall(
                r"\d",
                chunk,
            )
        )

        # Chu?i ngày d? 8 s?: 24031995.
        if len(digits) == 8:
            normalized_date = build_date(
                digits[0:2],
                digits[2:4],
                digits[4:8],
            )

            if normalized_date:
                return normalized_date

        # OCR th?a m?t ch? s?: 240311995.
        if len(digits) == 9:
            candidates: list[str] = []

            # Uu tiên xóa m?t kư t? trong c?p l?p.
            for index in range(
                len(digits) - 1
            ):
                if digits[index] == digits[index + 1]:
                    candidates.append(
                        digits[:index]
                        + digits[index + 1:]
                    )

            # Phuong án d? pḥng: th? xóa t?ng ch? s?.
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
    "Đă c?p nh?t normalize_date phiên b?n 3."
)
