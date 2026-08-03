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
    "HO", "VA", "TEN", "FULL", "NAME", "NO", "SO",
    "IDENTITY", "CARD", "CITIZEN", "NGAY", "SINH",
    "DATE", "OF", "BIRTH", "GIOI", "TINH", "SEX",
    "QUOC", "TICH", "NATIONALITY",
}

STOP_LABEL_PATTERN = re.compile(
    r"\b("
    r"ho\s*va\s*ten|full\s*name|"
    r"ngay\s*sinh|date\s*of\s*birth|"
    r"gioi\s*tinh|sex|"
    r"quoc\s*tich|nationality|"
    r"que\s*quan|place\s*of\s*origin|"
    r"noi\s*thuong\s*tru|place\s*of\s*residence|"
    r"(?:co|c[o0])\s*[g9]ia\s*(?:tr[iyj1l]|t[1il])\s*den|date\s*of\s*expiry"
    r")\b",
    flags=re.IGNORECASE,
)


def remove_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFD", str(value))
    text = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )
    return text.replace("đ", "d").replace("Đ", "D")


def normalize_spaces(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = re.sub(r"\s+", " ", str(value)).strip()
    return normalized or None


def normalize_ocr_digits(value: str) -> str:
    return str(value).translate(
        str.maketrans({
            "O": "0", "o": "0",
            "I": "1", "l": "1", "|": "1",
        })
    )


def normalize_date(value: str | None) -> str | None:
    if value is None:
        return None

    text = normalize_ocr_digits(value)
    text = re.sub(r"\s+", "", text)

    def build_date(digits: str) -> str | None:
        if len(digits) != 8:
            return None

        candidate = f"{digits[:2]}/{digits[2:4]}/{digits[4:]}"
        try:
            parsed = datetime.strptime(candidate, "%d/%m/%Y")
        except ValueError:
            return None

        if not 1900 <= parsed.year <= 2100:
            return None
        return candidate

    match = re.search(
        r"(?<!\d)(\d{1,2})[./\\-](\d{1,2})[./\\-](\d{4})(?!\d)",
        text,
    )
    if match:
        day, month, year = match.groups()
        result = build_date(
            f"{int(day):02d}{int(month):02d}{year}"
        )
        if result:
            return result

    digits = "".join(re.findall(r"\d", text))

    if len(digits) == 8:
        return build_date(digits)

    if len(digits) == 9:
        candidates: list[str] = []

        for index in range(8):
            if digits[index] == digits[index + 1]:
                candidates.append(
                    digits[:index] + digits[index + 1:]
                )

        for index in range(9):
            candidate = digits[:index] + digits[index + 1:]
            if candidate not in candidates:
                candidates.append(candidate)

        for candidate in candidates:
            result = build_date(candidate)
            if result:
                return result

    return None


def normalize_id_number(value: str | None) -> str | None:
    if not value:
        return None

    digits = re.sub(r"\D", "", normalize_ocr_digits(value))
    match = re.search(r"\d{12}", digits)
    return match.group(0) if match else None


def _plain(value: str | None) -> str:
    if not value:
        return ""
    text = remove_accents(value).lower()
    text = re.sub(r"[^a-z0-9/,;:\-\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _truncate_at_next_label(value: str) -> str:
    match = STOP_LABEL_PATTERN.search(value)
    if not match:
        return value
    return value[:match.start()]


def normalize_full_name(value: str | None) -> str | None:
    if not value:
        return None

    text = remove_accents(value).upper()
    text = _truncate_at_next_label(text)
    text = re.sub(r"[^A-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = [
        word for word in text.split()
        if word not in INVALID_NAME_WORDS
        and len(word) > 1
    ]

    if not 2 <= len(words) <= 7:
        return None

    return " ".join(words)


def normalize_nationality(value: str | None) -> str | None:
    if not value:
        return None

    compact = re.sub(r"[^a-z]", "", _plain(value))
    if "vietnam" in compact or "vietnan" in compact:
        return "Viet Nam"
    return None


def normalize_gender(value: str | None) -> str | None:
    """
    Chuẩn hóa giới tính CCCD.

    Quy ước:
        Nam -> Nam
        Nữ  -> Nu

    Chấp nhận các lỗi OCR thường gặp:
        Nu, Nữ, Ni, Nw, Nv, Nii -> Nu

    Không nhận "Nam" trong "Viet Nam".
    """

    if not value:
        return None

    text = remove_accents(str(value))
    text = text.lower().strip()

    # Chỉ giữ chữ cái
    text = re.sub(r"[^a-z]", "", text)

    # Giá trị rỗng
    if not text:
        return None

    # ====== NỮ ======
    if text in {
        "nu",
        "female",
        "ni",
        "nw",
        "nv",
        "nii",
        "nuu",
    }:
        return "Nu"

    # ====== NAM ======
    # Chỉ chấp nhận đúng "nam", không dùng substring.
    if text in {
        "nam",
        "male",
    }:
        return "Nam"

    return None

def _extract_after_label(
    line: str,
    label_pattern: str,
) -> str:
    match = re.search(label_pattern, line, flags=re.IGNORECASE)
    if not match:
        return ""

    suffix = line[match.end():]
    suffix = _truncate_at_next_label(suffix)
    return suffix.strip(" :;,/-")


def _extract_before_label(
    line: str,
    label_pattern: str,
) -> str:
    match = re.search(label_pattern, line, flags=re.IGNORECASE)
    if not match:
        return ""

    prefix = line[:match.start()]
    return prefix.strip(" :;,/-")


def recover_full_name(raw_text: list[str] | None) -> str | None:
    lines = [str(line) for line in raw_text or [] if line]
    label_pattern = r"(?:ho\s*va\s*ten\s*/?\s*|full\s*name\s*:?)"

    for index, line in enumerate(lines):
        plain_line = remove_accents(line)
        match = re.search(label_pattern, plain_line, flags=re.IGNORECASE)
        if not match:
            continue

        candidates = [
            plain_line[:match.start()],
            plain_line[match.end():],
        ]
        if index > 0:
            candidates.append(remove_accents(lines[index - 1]))
        candidates.extend(remove_accents(item) for item in lines[index + 1:index + 3])

        for value in candidates:
            candidate = normalize_full_name(value)
            if candidate:
                return candidate

    return None

def recover_gender(
    raw_text: list[str] | None,
) -> str | None:
    """
    Phục hồi giới tính từ đúng dòng chứa nhãn Giới tính / Sex.

    Ví dụ:
        Gioi tinh / Sex: Nu Quoc tich / Nationality: Viet Nam

    Kết quả:
        Nu

    Phần Quốc tịch bị cắt bỏ trước khi chuẩn hóa để chữ
    "Nam" trong "Viet Nam" không gây nhận sai.
    """

    if not raw_text:
        return None

    for raw_line in raw_text:
        if not raw_line:
            continue

        line = remove_accents(
            str(raw_line)
        )

        normalized_line = re.sub(
            r"\s+",
            " ",
            line,
        ).strip()

        # Chỉ xét các dòng có nhãn giới tính.
        if not re.search(
            r"\b(?:gioi\s*tinh|sex)\b",
            normalized_line,
            flags=re.IGNORECASE,
        ):
            continue

        # Cắt bỏ toàn bộ phần quốc tịch phía sau.
        gender_section = re.split(
            r"\b(?:quoc\s*tich|nationality)\b",
            normalized_line,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]

        # Lấy nội dung sau nhãn Sex: nếu có.
        sex_match = re.search(
            r"\bsex\b\s*[:/\-]?\s*(.*)$",
            gender_section,
            flags=re.IGNORECASE,
        )

        if sex_match:
            gender_value = sex_match.group(1)
        else:
            # Fallback lấy sau nhãn Giới tính.
            gender_match = re.search(
                r"\bgioi\s*tinh\b"
                r"\s*[:/\-]?\s*"
                r"(.*)$",
                gender_section,
                flags=re.IGNORECASE,
            )

            if not gender_match:
                continue

            gender_value = gender_match.group(1)

        # Xóa nhãn Sex còn sót nếu OCR ghép dạng:
        # Gioi tinh / Sex: Nu
        gender_value = re.sub(
            r"^\s*/?\s*sex\s*[:/\-]?\s*",
            "",
            gender_value,
            flags=re.IGNORECASE,
        )

        gender = normalize_gender(
            gender_value
        )

        if gender:
            return gender

    return None


def recover_nationality(raw_text: list[str] | None) -> str | None:
    label_pattern = r"(?:quoc\s*tich\s*/?\s*|nationality\s*:?)"

    for line in raw_text or []:
        plain_line = remove_accents(str(line))

        if not re.search(label_pattern, plain_line, flags=re.IGNORECASE):
            continue

        value = _extract_after_label(plain_line, label_pattern)
        nationality = normalize_nationality(value)

        if nationality:
            return nationality

    return None


def recover_labeled_date(
    raw_text: list[str] | None,
    field_name: str,
) -> str | None:
    """
    Phục hồi ngày từ đúng dòng chứa nhãn.

    Hỗ trợ dòng có đồng thời nhãn tiếng Việt và tiếng Anh, ví dụ:
        Ngay sinh / Date of birth: 24/0311995
    """

    patterns = {
        "dateOfBirth": (
            r"(?:ngay\s*sinh|date\s*of\s*birth)"
        ),
        "dateOfExpiry": (
            r"(?:(?:co|c[o0])\s*[g9]ia\s*(?:tr[iyj1l]|t[1il])\s*den"
            r"|date\s*of\s*expiry)"
        ),
    }

    label_pattern = patterns.get(field_name)
    if not label_pattern:
        return None

    for line in raw_text or []:
        plain_line = remove_accents(str(line))

        matches = list(
            re.finditer(
                label_pattern,
                plain_line,
                flags=re.IGNORECASE,
            )
        )

        if not matches:
            continue

        # Ưu tiên phần sau nhãn cuối cùng trên dòng.
        for match in reversed(matches):
            value = plain_line[match.end():]
            value = value.strip(" :;,/-")

            date = normalize_date(value)
            if date:
                return date

        # Fallback cho các dạng:
        # 24/0311995, 24031995, 24-03-1995, 24.03.1995.
        date = normalize_date(plain_line)
        if date:
            return date

    return None


def _clean_address_text(value: str | None) -> str | None:
    if not value:
        return None

    text = remove_accents(value)

    # Loại cụm ngày hết hạn OCR sai nhẹ nhưng giữ phần địa chỉ phía sau.
    text = re.sub(
        r"(?:co|c[o0])\s*[g9]ia\s*(?:tr[iyj1l]|t[1il])\s*den\s*"
        r"\d{1,2}[./-]\d{1,2}[./-]\d{4}",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\b(?:que\s*quan|place\s*of\s*origin|"
        r"noi\s*thuong\s*tru|place\s*of\s*residence|"
        r"(?:co|c[o0])\s*[g9]ia\s*(?:tr[iyj1l]|t[1il])\s*den|date\s*of\s*expiry)"
        r"\b\s*:?",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{4}\b",
        " ",
        text,
    )
    text = text.replace(";", ",")
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r",(?:\s*,)+", ", ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" ,/:;-")

    return text or None


def _is_expiry_line(value: str) -> bool:
    """Nhận diện dòng hạn sử dụng có thể chen giữa hai dòng địa chỉ."""
    return bool(
        re.search(
            r"(?:co|c[o0])\s*[g9]ia\s*(?:tr[iyj1l]|t[1il])\s*den"
            r"|date\s*of\s*expiry",
            value,
            flags=re.IGNORECASE,
        )
    )


def recover_address(
    raw_text: list[str] | None,
    field_name: str,
) -> str | None:
    lines = [remove_accents(str(line)) for line in raw_text or [] if line]

    if field_name == "placeOfOrigin":
        label_pattern = (
            r"(?:que\s*quan\s*/?\s*(?:place\s*of\s*origin)?"
            r"|place\s*of\s*origin)\s*:?"
        )
        stop_pattern = r"\b(?:noi\s*thuong\s*tru|place\s*of\s*residence|date\s*of\s*expiry)\b"
        max_pieces = 2
    else:
        label_pattern = (
            r"(?:noi\s*thuong\s*tru\s*/?\s*(?:place\s*of\s*residence)?"
            r"|place\s*of\s*residence)\s*:?"
        )
        stop_pattern = r"\b(?:date\s*of\s*expiry)\b"
        max_pieces = 3

    for index, line in enumerate(lines):
        label_match = re.search(label_pattern, line, flags=re.IGNORECASE)
        if not label_match:
            continue

        pieces: list[str] = []
        before = line[:label_match.start()].strip(" :;,/-\"")
        after = line[label_match.end():].strip(" :;,/-\"")

        def usable(piece: str) -> bool:
            cleaned = _clean_address_text(piece)
            if not cleaned:
                return False
            tokens = re.findall(r"[a-z0-9]+", _plain(cleaned))
            return len(tokens) >= 2 and not all(len(token) == 1 for token in tokens)

        if usable(before):
            pieces.append(before)
        if usable(after):
            pieces.append(after)

        for next_line in lines[index + 1:index + 4]:
            # EasyOCR sắp xếp theo tọa độ dọc. Trên CCCD, dòng
            # "Có giá trị đến" ở bên trái có thể nằm giữa hai dòng
            # nơi thường trú ở bên phải. Bỏ qua riêng dòng hạn sử dụng
            # và tiếp tục tìm phần địa chỉ còn lại trong cửa sổ hiện tại.
            if (
                field_name == "placeOfResidence"
                and _is_expiry_line(next_line)
            ):
                continue

            if re.search(stop_pattern, next_line, flags=re.IGNORECASE):
                break
            if STOP_LABEL_PATTERN.search(next_line):
                break
            if usable(next_line):
                pieces.append(next_line.strip(" :;,/-\""))
            if len(pieces) >= max_pieces:
                break

        candidate = _clean_address_text(", ".join(pieces))
        if candidate and is_valid_address(candidate, field_name):
            return candidate

    return None

def is_valid_address(
    value: str | None,
    field_name: str,
) -> bool:
    if not value:
        return False

    text = _plain(value)
    tokens = re.findall(r"[a-z0-9]+", text)

    if len(tokens) < 3:
        return False

    single_tokens = [token for token in tokens if len(token) == 1]
    if len(single_tokens) >= 3:
        return False

    forbidden = (
        "full name", "date of birth", "gioi tinh", "sex",
        "quoc tich", "nationality", "date of expiry",
        "co gia tri den",
    )
    if any(label in text for label in forbidden):
        return False

    if field_name == "placeOfOrigin":
        if sum(character.isdigit() for character in text) > 2:
            return False

    noise_words = {
        "fiaco", "onyu", "notmuong", "uliesicetce",
        "deleiabexpiry", "codauden", "ogiatno1n",
        "fresidencethon", "thackcks", "thackho",
    }
    if sum(token in noise_words for token in tokens) >= 1:
        return False

    return True


def select_address(
    field_name: str,
    field_value: str | None,
    full_card_value: str | None,
    raw_text: list[str] | None,
) -> tuple[str | None, str]:
    raw_value = recover_address(raw_text, field_name)
    full_value = _clean_address_text(full_card_value)
    field_clean = _clean_address_text(field_value)

    if raw_value and is_valid_address(raw_value, field_name):
        return raw_value, "RAW_TEXT_RECOVERY"

    if full_value and is_valid_address(full_value, field_name):
        return full_value, "FULL_CARD_OCR"

    if field_clean and is_valid_address(field_clean, field_name):
        return field_clean, "FIELD_OCR"

    return None, "NOT_FOUND"


def select_full_name(
    field_value: str | None,
    full_card_value: str | None,
    raw_text: list[str] | None,
) -> tuple[str | None, str]:
    """
    Chọn họ tên theo thứ tự ưu tiên:

    1. FULL_CARD_OCR nếu hợp lệ.
    2. RAW_TEXT_RECOVERY nếu full-card không có hoặc không hợp lệ.
    3. FIELD_OCR nếu hai nguồn trên không dùng được.

    Thứ tự này giữ đúng provenance của dữ liệu:
    khi full_card_data đã có họ tên hợp lệ thì không gắn nhãn
    RAW_TEXT_RECOVERY chỉ vì raw_text cũng đọc ra cùng giá trị.
    """

    full_name = normalize_full_name(
        full_card_value
    )
    if full_name:
        return full_name, "FULL_CARD_OCR"

    raw_name = recover_full_name(
        raw_text
    )
    if raw_name:
        return raw_name, "RAW_TEXT_RECOVERY"

    field_name = normalize_full_name(
        field_value
    )
    if field_name:
        return field_name, "FIELD_OCR"

    return None, "NOT_FOUND"


def select_gender(
    field_value: str | None,
    full_card_value: str | None,
    raw_text: list[str] | None,
) -> tuple[str | None, str]:
    raw_gender = recover_gender(raw_text)
    if raw_gender:
        return raw_gender, "RAW_TEXT_RECOVERY"

    full_gender = normalize_gender(full_card_value)
    if full_gender:
        return full_gender, "FULL_CARD_OCR"

    field_gender = normalize_gender(field_value)
    if field_gender:
        return field_gender, "FIELD_OCR"

    return None, "NOT_FOUND"


def select_nationality(
    field_value: str | None,
    full_card_value: str | None,
    raw_text: list[str] | None,
) -> tuple[str | None, str]:
    raw_value = recover_nationality(raw_text)
    if raw_value:
        return raw_value, "RAW_TEXT_RECOVERY"

    full_value = normalize_nationality(full_card_value)
    if full_value:
        return full_value, "FULL_CARD_OCR"

    field_value_normalized = normalize_nationality(field_value)
    if field_value_normalized:
        return field_value_normalized, "FIELD_OCR"

    return None, "NOT_FOUND"


def select_date(
    field_name: str,
    field_value: str | None,
    full_card_value: str | None,
    raw_text: list[str] | None,
) -> tuple[str | None, str]:
    raw_value = recover_labeled_date(raw_text, field_name)
    if raw_value:
        return raw_value, "RAW_TEXT_RECOVERY"

    full_value = normalize_date(full_card_value)
    if full_value:
        return full_value, "FULL_CARD_OCR"

    field_value_normalized = normalize_date(field_value)
    if field_value_normalized:
        return field_value_normalized, "FIELD_OCR"

    return None, "NOT_FOUND"


def fuse_ocr_data(
    full_card_data: dict[str, Any] | None,
    field_data: dict[str, Any] | None,
    raw_text: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    full_card = full_card_data or {}
    fields = field_data or {}

    result: dict[str, Any] = {}
    sources: dict[str, str] = {}

    for field_name in FIELD_NAMES:
        result[field_name] = deepcopy(
            full_card.get(field_name)
            or fields.get(field_name)
        )
        sources[field_name] = (
            "FULL_CARD_OCR"
            if full_card.get(field_name)
            else "FIELD_OCR"
            if fields.get(field_name)
            else "NOT_FOUND"
        )

    id_number = (
        normalize_id_number(full_card.get("idNumber"))
        or normalize_id_number(fields.get("idNumber"))
    )
    result["idNumber"] = id_number
    sources["idNumber"] = (
        "FULL_CARD_OCR"
        if normalize_id_number(full_card.get("idNumber"))
        else "FIELD_OCR"
        if normalize_id_number(fields.get("idNumber"))
        else "NOT_FOUND"
    )

    full_name, source = select_full_name(
        fields.get("fullName"),
        full_card.get("fullName"),
        raw_text,
    )
    result["fullName"] = full_name
    sources["fullName"] = source

    date_of_birth, source = select_date(
        "dateOfBirth",
        fields.get("dateOfBirth"),
        full_card.get("dateOfBirth"),
        raw_text,
    )
    result["dateOfBirth"] = date_of_birth
    sources["dateOfBirth"] = source

    gender, source = select_gender(
        fields.get("gender"),
        full_card.get("gender"),
        raw_text,
    )
    result["gender"] = gender
    sources["gender"] = source

    nationality, source = select_nationality(
        fields.get("nationality"),
        full_card.get("nationality"),
        raw_text,
    )
    result["nationality"] = nationality
    sources["nationality"] = source

    for field_name in ("placeOfOrigin", "placeOfResidence"):
        value, source = select_address(
            field_name,
            fields.get(field_name),
            full_card.get(field_name),
            raw_text,
        )
        result[field_name] = value
        sources[field_name] = source

    expiry, source = select_date(
        "dateOfExpiry",
        fields.get("dateOfExpiry"),
        full_card.get("dateOfExpiry"),
        raw_text,
    )
    result["dateOfExpiry"] = expiry
    sources["dateOfExpiry"] = source

    return result, sources