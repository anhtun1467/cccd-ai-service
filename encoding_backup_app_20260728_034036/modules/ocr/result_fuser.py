from __future__ import annotations

import re
import unicodedata
from copy import deepcopy
from datetime import datetime
from typing import Any


FIELD_NAMES = (
    "idNumber",
    "fullName",
    "dateOfBirth",
    "gender",
    "nationality",
    "placeOfOrigin",
    "placeOfResidence",
    "dateOfExpiry",
)


INVALID_NAME_WORDS = {
    "HO",
    "VA",
    "TEN",
    "FULL",
    "NAME",
    "NO",
    "SO",
    "IDENTITY",
    "CARD",
    "CITIZEN",
}


def remove_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value)

    return "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )


def normalize_spaces(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = re.sub(r"\s+", " ", str(value)).strip()

    return normalized or None


def normalize_ocr_digits(value: str) -> str:
    translation_table = str.maketrans(
        {
            "O": "0",
            "o": "0",
            "I": "1",
            "l": "1",
            "|": "1",
        }
    )

    return value.translate(translation_table)


def normalize_date(value: str | None) -> str | None:
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


def is_valid_full_name(value: str | None) -> bool:
    if not value:
        return False

    normalized = remove_accents(value).upper()
    normalized = re.sub(r"[^A-Z\s]", " ", normalized)
    words = normalized.split()

    if not 2 <= len(words) <= 8:
        return False

    if any(word.isdigit() for word in words):
        return False

    meaningful_words = [
        word
        for word in words
        if word not in INVALID_NAME_WORDS
    ]

    # Một họ tên hợp lệ phải còn tối thiểu hai từ có nghĩa
    # sau khi loại các nhãn như HO VA TEN, FULL NAME.
    return len(meaningful_words) >= 2


def normalize_full_name(value: str | None) -> str | None:
    if not is_valid_full_name(value):
        return None

    normalized = normalize_spaces(value)

    if normalized is None:
        return None

    normalized = re.sub(
        r"[^A-Za-zÀ-ỹĐđ\s]",
        " ",
        normalized,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()

    return normalized.upper() or None


def normalize_nationality(
    value: str | None,
) -> str | None:
    if not value:
        return None

    normalized = remove_accents(value).lower()
    normalized = re.sub(
        r"[^a-z\s]",
        " ",
        normalized,
    )
    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()

    compact = normalized.replace(" ", "")

    if (
        "vietnam" in compact
        or "vietnan" in compact
        or compact in {"vn", "vnm"}
    ):
        return "Viet Nam"

    # Không chấp nhận các token OCR vô nghĩa như "Mre".
    if len(normalized) < 5:
        return None

    return value.strip()


def select_full_name(
    field_value: str | None,
    full_card_value: str | None,
) -> tuple[str | None, str]:
    field_name = normalize_full_name(field_value)
    full_card_name = normalize_full_name(full_card_value)

    if field_name:
        return field_name, "FIELD_OCR"

    if full_card_name:
        return full_card_name, "FULL_CARD_OCR"

    return None, "NOT_FOUND"


def select_date(
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


def select_nationality(
    field_value: str | None,
    full_card_value: str | None,
    raw_text: list[str] | None,
) -> tuple[str | None, str]:
    field_nationality = normalize_nationality(
        field_value
    )

    if field_nationality:
        return field_nationality, "FIELD_OCR"

    full_card_nationality = normalize_nationality(
        full_card_value
    )

    if full_card_nationality:
        return (
            full_card_nationality,
            "FULL_CARD_OCR",
        )

    for line in raw_text or []:
        raw_nationality = normalize_nationality(line)

        if raw_nationality == "Viet Nam":
            return "Viet Nam", "RAW_TEXT_RECOVERY"

    return None, "NOT_FOUND"


def fuse_ocr_data(
    full_card_data: dict[str, Any] | None,
    field_data: dict[str, Any] | None,
    raw_text: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """
    Hợp nhất kết quả OCR toàn thẻ và OCR theo vùng.

    Nguyên tắc:
        - Chỉ dùng field OCR khi giá trị hợp lệ.
        - Nếu field OCR sai, fallback sang full-card OCR.
        - Ngày sinh có thể được phục hồi từ raw text.
    """

    full_card = full_card_data or {}
    fields = field_data or {}

    result: dict[str, Any] = {}
    sources: dict[str, str] = {}

    for field_name in FIELD_NAMES:
        result[field_name] = deepcopy(
            fields.get(field_name)
            or full_card.get(field_name)
        )

        if fields.get(field_name):
            sources[field_name] = "FIELD_OCR"
        elif full_card.get(field_name):
            sources[field_name] = "FULL_CARD_OCR"
        else:
            sources[field_name] = "NOT_FOUND"

    full_name, full_name_source = select_full_name(
        fields.get("fullName"),
        full_card.get("fullName"),
    )

    result["fullName"] = full_name
    sources["fullName"] = full_name_source

    date_of_birth, date_source = select_date(
        fields.get("dateOfBirth"),
        full_card.get("dateOfBirth"),
        raw_text,
    )

    result["dateOfBirth"] = date_of_birth
    sources["dateOfBirth"] = date_source

    nationality, nationality_source = (
        select_nationality(
            fields.get("nationality"),
            full_card.get("nationality"),
            raw_text,
        )
    )

    result["nationality"] = nationality
    sources["nationality"] = nationality_source

    return result, sources

