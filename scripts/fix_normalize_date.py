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

new_function = r'''def normalize_date(value: str | None) -> str | None:
    """
    Chu?n hóa ngày t? k?t qu? OCR.

    Ví d?:
        24/03/1995  -> 24/03/1995
        24-03-1995  -> 24/03/1995
        24.03.1995  -> 24/03/1995
        24031995    -> 24/03/1995
        24/031995   -> 24/03/1995
        24/0311995  -> 24/03/1995

    Tru?ng h?p 24/0311995 có m?t ch? s? OCR b? l?p.
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
    # 1. Th? d?nh d?ng dă có d? d?u phân cách
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
    # 2. L?y t?t c? ch? s? t? chu?i OCR
    # --------------------------------------------------

    digits = "".join(
        re.findall(
            r"\d",
            text,
        )
    )

    # Ngày dă d? 8 ch? s?.
    if len(digits) == 8:
        return build_date(digits)

    # --------------------------------------------------
    # 3. OCR th?a m?t ch? s?, t?ng c?ng 9 ch? s?
    # --------------------------------------------------

    if len(digits) == 9:
        candidate_digit_strings: list[str] = []

        # Uu tiên lo?i b? m?t ch? s? trong c?p b? l?p.
        #
        # Ví d?:
        # 240311995
        #     ^^
        #
        # B? m?t s? 1:
        # 24031995
        for index in range(
            len(digits) - 1
        ):
            if digits[index] == digits[index + 1]:
                candidate_digit_strings.append(
                    digits[:index]
                    + digits[index + 1:]
                )

        # N?u không ph?i l?i l?p rơ ràng, th? lo?i b?
        # t?ng ch? s? và ki?m tra ngày h?p l?.
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
    # 4. T́m c?m 8 ho?c 9 ch? s? trong chu?i dài
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
    "Đă s?a hàm normalize_date thành công."
)
