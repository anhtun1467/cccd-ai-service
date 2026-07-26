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

new_function = r'''def normalize_date(value: str | None) -> str | None:
    """
    Chuẩn hóa ngày từ kết quả OCR.

    Ví dụ:
        24/03/1995  -> 24/03/1995
        24-03-1995  -> 24/03/1995
        24.03.1995  -> 24/03/1995
        24031995    -> 24/03/1995
        24/031995   -> 24/03/1995
        24/0311995  -> 24/03/1995

    Trường hợp 24/0311995 có một chữ số OCR bị lặp.
    """

    if value is None:
        return None

    text = normalize_ocr_digits(
        str(value)
    )

    text = re.sub(
        r"\s+",
        "",
        text,
    )

    def build_date(
        digits: str,
    ) -> str | None:
        if len(digits) != 8:
            return None

        day = digits[0:2]
        month = digits[2:4]
        year = digits[4:8]

        candidate = (
            f"{day}/{month}/{year}"
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

    # --------------------------------------------------
    # 1. Thử định dạng đã có đủ dấu phân cách
    # --------------------------------------------------

    normal_match = re.search(
        r"(?<!\d)"
        r"(\d{1,2})"
        r"[./\\-]"
        r"(\d{1,2})"
        r"[./\\-]"
        r"(\d{4})"
        r"(?!\d)",
        text,
    )

    if normal_match:
        day, month, year = normal_match.groups()

        normalized_digits = (
            f"{int(day):02d}"
            f"{int(month):02d}"
            f"{year}"
        )

        normalized_date = build_date(
            normalized_digits
        )

        if normalized_date:
            return normalized_date

    # --------------------------------------------------
    # 2. Lấy tất cả chữ số từ chuỗi OCR
    # --------------------------------------------------

    digits = "".join(
        re.findall(
            r"\d",
            text,
        )
    )

    # Ngày đã đủ 8 chữ số.
    if len(digits) == 8:
        return build_date(digits)

    # --------------------------------------------------
    # 3. OCR thừa một chữ số, tổng cộng 9 chữ số
    # --------------------------------------------------

    if len(digits) == 9:
        candidate_digit_strings: list[str] = []

        # Ưu tiên loại bỏ một chữ số trong cặp bị lặp.
        #
        # Ví dụ:
        # 240311995
        #     ^^
        #
        # Bỏ một số 1:
        # 24031995
        for index in range(
            len(digits) - 1
        ):
            if digits[index] == digits[index + 1]:
                candidate_digit_strings.append(
                    digits[:index]
                    + digits[index + 1:]
                )

        # Nếu không phải lỗi lặp rõ ràng, thử loại bỏ
        # từng chữ số và kiểm tra ngày hợp lệ.
        for index in range(len(digits)):
            candidate = (
                digits[:index]
                + digits[index + 1:]
            )

            if candidate not in candidate_digit_strings:
                candidate_digit_strings.append(
                    candidate
                )

        for candidate_digits in candidate_digit_strings:
            normalized_date = build_date(
                candidate_digits
            )

            if normalized_date:
                return normalized_date

    # --------------------------------------------------
    # 4. Tìm cụm 8 hoặc 9 chữ số trong chuỗi dài
    # --------------------------------------------------

    digit_sequences = re.findall(
        r"\d{8,9}",
        text,
    )

    for sequence in digit_sequences:
        if len(sequence) == 8:
            normalized_date = build_date(
                sequence
            )

            if normalized_date:
                return normalized_date

        if len(sequence) == 9:
            for index in range(
                len(sequence) - 1
            ):
                if (
                    sequence[index]
                    == sequence[index + 1]
                ):
                    candidate = (
                        sequence[:index]
                        + sequence[index + 1:]
                    )

                    normalized_date = build_date(
                        candidate
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
    "Đã sửa hàm normalize_date thành công."
)
