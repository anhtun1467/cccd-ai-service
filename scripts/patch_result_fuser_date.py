from pathlib import Path


file_path = Path(
    "app/modules/ocr/result_fuser.py"
)

content = file_path.read_text(
    encoding="utf-8-sig"
)

start_marker = "def normalize_date("
end_marker = "\ndef is_valid_full_name("

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

new_function = '''def normalize_date(value: str | None) -> str | None:
    """
    Chu?n hóa ngày tháng t? k?t qu? OCR.

    Các ví d? du?c h? tr?:
        24/03/1995   -> 24/03/1995
        24/0311995   -> 24/03/1995
        2403/1995    -> 24/03/1995
        24031995     -> 24/03/1995
        24-03-1995   -> 24/03/1995
        24.03.1995   -> 24/03/1995
    """

    if not value:
        return None

    text = normalize_ocr_digits(
        str(value)
    )

    text = re.sub(
        r"\\s+",
        "",
        text,
    )

    candidates: list[
        tuple[str, str, str]
    ] = []

    patterns = (
        # Chu?n: 24/03/1995
        r"(?<!\\d)"
        r"(\\d{1,2})"
        r"[./\\\\-]"
        r"(\\d{1,2})"
        r"[./\\\\-]"
        r"(\\d{4})"
        r"(?!\\d)",

        # M?t d?u gi?a tháng và nam:
        # 24/0311995
        r"(?<!\\d)"
        r"(\\d{2})"
        r"[./\\\\-]"
        r"(\\d{2})"
        r"(\\d{4})"
        r"(?!\\d)",

        # M?t d?u gi?a ngày và tháng:
        # 2403/1995
        r"(?<!\\d)"
        r"(\\d{2})"
        r"(\\d{2})"
        r"[./\\\\-]"
        r"(\\d{4})"
        r"(?!\\d)",

        # M?t toàn b? d?u phân cách:
        # 24031995
        r"(?<!\\d)"
        r"(\\d{2})"
        r"(\\d{2})"
        r"(\\d{4})"
        r"(?!\\d)",
    )

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
        )

        if match:
            candidates.append(
                match.groups()
            )

    # Phuong án d? pḥng: l?y chu?i s? g?n ngày sinh.
    # X? lư tru?ng h?p OCR t?o ra 9 ch? s? nhu 240311995.
    digit_sequences = re.findall(
        r"\\d+",
        text,
    )

    for digit_sequence in digit_sequences:
        if len(digit_sequence) == 8:
            candidates.append(
                (
                    digit_sequence[0:2],
                    digit_sequence[2:4],
                    digit_sequence[4:8],
                )
            )

        elif len(digit_sequence) == 9:
            # M?t s? OCR có th? chèn du m?t ch? s?.
            # Không t? doán tùy ti?n n?u chua th? t?o ngày h?p l?.
            possible_values = (
                digit_sequence[:8],
                digit_sequence[1:9],
            )

            for possible_value in possible_values:
                candidates.append(
                    (
                        possible_value[0:2],
                        possible_value[2:4],
                        possible_value[4:8],
                    )
                )

    checked_candidates: set[str] = set()

    for day, month, year in candidates:
        try:
            candidate = (
                f"{int(day):02d}/"
                f"{int(month):02d}/"
                f"{int(year):04d}"
            )
        except ValueError:
            continue

        if candidate in checked_candidates:
            continue

        checked_candidates.add(candidate)

        try:
            parsed_date = datetime.strptime(
                candidate,
                "%d/%m/%Y",
            )
        except ValueError:
            continue

        if not 1900 <= parsed_date.year <= 2100:
            continue

        return candidate

    return None

'''

updated_content = (
    content[:start_index]
    + new_function
    + content[end_index + 1:]
)

file_path.write_text(
    updated_content,
    encoding="utf-8",
)

print(
    "Đă c?p nh?t normalize_date thành công."
)
