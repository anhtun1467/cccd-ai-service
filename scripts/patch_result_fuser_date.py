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
        "Không tìm thấy hàm normalize_date."
    )

if end_index == -1:
    raise RuntimeError(
        "Không tìm thấy hàm is_valid_full_name."
    )

new_function = '''def normalize_date(value: str | None) -> str | None:
    """
    Chuẩn hóa ngày tháng từ kết quả OCR.

    Các ví dụ được hỗ trợ:
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
        # Chuẩn: 24/03/1995
        r"(?<!\\d)"
        r"(\\d{1,2})"
        r"[./\\\\-]"
        r"(\\d{1,2})"
        r"[./\\\\-]"
        r"(\\d{4})"
        r"(?!\\d)",

        # Mất dấu giữa tháng và năm:
        # 24/0311995
        r"(?<!\\d)"
        r"(\\d{2})"
        r"[./\\\\-]"
        r"(\\d{2})"
        r"(\\d{4})"
        r"(?!\\d)",

        # Mất dấu giữa ngày và tháng:
        # 2403/1995
        r"(?<!\\d)"
        r"(\\d{2})"
        r"(\\d{2})"
        r"[./\\\\-]"
        r"(\\d{4})"
        r"(?!\\d)",

        # Mất toàn bộ dấu phân cách:
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

    # Phương án dự phòng: lấy chuỗi số gần ngày sinh.
    # Xử lý trường hợp OCR tạo ra 9 chữ số như 240311995.
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
            # Một số OCR có thể chèn dư một chữ số.
            # Không tự đoán tùy tiện nếu chưa thể tạo ngày hợp lệ.
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
    "Đã cập nhật normalize_date thành công."
)
