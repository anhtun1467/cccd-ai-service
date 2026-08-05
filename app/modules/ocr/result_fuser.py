from __future__ import annotations

import re
import unicodedata
from copy import deepcopy
from datetime import datetime
from difflib import SequenceMatcher
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
    "VA", "TEN", "FULL", "NAME", "NO", "SO",
    "IDENTITY", "CARD", "CITIZEN", "NGAY", "SINH",
    "DATE", "OF", "BIRTH", "GIOI", "TINH", "SEX",
    "QUOC", "TICH", "NATIONALITY",
}

ORIGIN_EN_LABEL_PATTERN = (
    r"place\s*o[fl]\s*o(?:ri|r|n|ni|i)?g(?:i)?n"
)
RESIDENCE_EN_LABEL_PATTERN = (
    r"place\s*o[fl]\s*['\"]?\s*resid[eaou]nce"
)
RESIDENCE_VI_LABEL_PATTERN = (
    r"n[o0](?:i|[1l])?\s*thu[o0]ng\s*tr?u"
)
EXPIRY_LABEL_PATTERN = (
    r"(?:co|c[o0])\s*[i1l]?\s*[g9]ia(?:\s+[a-z])?\s*[\(\[]?\s*"
    r"(?:tr[iyj1l]|t[1il])\s*den|date\s*of\s*expiry"
)

STOP_LABEL_PATTERN = re.compile(
    rf"\b("
    r"ho\s*va\s*ten|full\s*name|"
    r"ngay\s*sinh|date\s*of\s*birth|"
    r"gioi\s*tinh|sex|"
    r"quoc\s*tich|nationality|"
    rf"que\s*quan|{ORIGIN_EN_LABEL_PATTERN}|"
    rf"{RESIDENCE_VI_LABEL_PATTERN}|{RESIDENCE_EN_LABEL_PATTERN}|"
    rf"{EXPIRY_LABEL_PATTERN}"
    r")",
    flags=re.IGNORECASE,
)


KNOWN_ADMINISTRATIVE_PHRASES: tuple[tuple[str, str], ...] = (
    # So khớp bằng bản không dấu để sửa dấu thanh OCR đọc thiếu/sai.
    # Cụm dài phải đứng trước cụm ngắn để tránh thay từng phần.
    (r"\bthua\s+thien\s+hue\b", "Thừa Thiên Huế"),
    (r"\bthanh\s+pho\s+thanh\s+hoa\b", "Thành phố Thanh Hóa"),
    (r"\bthi\s+tran\s+voi\b", "Thị trấn Vôi"),
    (r"\bphu\s+van\s+nam\b", "Phú Vân Nam"),
    (r"\bbach\s+thuan\b", "Bách Thuận"),
    (r"\bvu\s+thu\b", "Vũ Thư"),
    (r"\bthai\s+binh\b", "Thái Bình"),
    (r"\bhai\s+chau\b", "Hải Châu"),
    (r"\bhai\s+hau\b", "Hải Hậu"),
    (r"\bnam\s+dinh\b", "Nam Định"),
    (r"\bdong\s+son\b", "Đông Sơn"),
    (r"\bnguyen\s+hong\b", "Nguyên Hồng"),
    (r"\btan\s+son\b", "Tân Sơn"),
    (r"\bthanh\s+hoa\b", "Thanh Hóa"),
    (r"\bea\s+hiao\b", "Ea Hiao"),
    (r"\bea\s+h\s*['’]?\s*leo\b", "Ea H'Leo"),
    (r"\bdak\s+lak\b", "Đắk Lắk"),
    (r"\ble\s+loi\b", "Lê Lợi"),
    (r"\bpho\s+voi\b", "Phố Vôi"),
    (r"\blang\s+giang\b", "Lạng Giang"),
    (r"\bbac\s+giang\b", "Bắc Giang"),
    (r"\bvinh\s+ha\b", "Vinh Hà"),
    (r"\bphu\s+vang\b", "Phú Vang"),
    (r"\bphu\s+tuyen\b", "Phú Tuyên"),
    (r"\bbinh\s+thanh\b", "Bình Thành"),
    (r"\bhuong\s+tra\b", "Hương Trà"),
    (r"\bthanh\s+pho\b", "Thành phố"),
    (r"\bthi\s+tran\b", "Thị trấn"),
)


CANONICAL_DISPUTED_SURNAMES = {
    # Chỉ áp dụng khi ít nhất hai nguồn OCR cùng đọc được họ "HO"
    # nhưng cho dấu khác nhau (ví dụ HỔ và HÔ). Trường hợp chỉ có một
    # nguồn sẽ được giữ nguyên để không tự đoán tên người dùng.
    "HO": "HỒ",
}


# Chỉ áp dụng cho từ đầu tiên của họ tên khi đã xác định được giới tính.
# Đây là các họ phổ biến, không dùng như một từ điển đoán mọi âm tiết.
COMMON_VIETNAMESE_SURNAMES = {
    "NGUYEN": "NGUYỄN",
    "TRAN": "TRẦN",
    "LE": "LÊ",
    "PHAM": "PHẠM",
    "HOANG": "HOÀNG",
    "HUYNH": "HUỲNH",
    "PHAN": "PHAN",
    "VU": "VŨ",
    "VO": "VÕ",
    "DANG": "ĐẶNG",
    "BUI": "BÙI",
    "DO": "ĐỖ",
    "HO": "HỒ",
    "NGO": "NGÔ",
    "DUONG": "DƯƠNG",
    "LY": "LÝ",
    "TRUONG": "TRƯƠNG",
    "DINH": "ĐINH",
    "TRINH": "TRỊNH",
    "DAO": "ĐÀO",
    "LUU": "LƯU",
    "TA": "TẠ",
}

FEMALE_NAME_DIACRITICS = {
    "THI": "THỊ",
    "MAY": "MÂY",
    "HUYEN": "HUYỀN",
}

MALE_NAME_DIACRITICS = {
    "VAN": "VĂN",
    "TUNG": "TÙNG",
}


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
    # remove_accents() giữ nguyên số lượng ký tự, vì vậy có thể dùng vị
    # trí match trên bản không dấu để cắt chuỗi gốc mà không làm mất dấu.
    match = STOP_LABEL_PATTERN.search(remove_accents(value))
    if not match:
        return value
    return value[:match.start()]


def normalize_full_name(value: str | None) -> str | None:
    if not value:
        return None

    text = _truncate_at_next_label(str(value))
    raw_words = re.findall(
        r"[^\W\d_]+",
        text,
        flags=re.UNICODE,
    )
    words = []

    for word in raw_words:
        plain_word = remove_accents(word).upper()
        if (
            plain_word in INVALID_NAME_WORDS
            or len(plain_word) <= 1
        ):
            continue
        words.append(word.upper())

    if not 2 <= len(words) <= 7:
        return None

    return " ".join(words)


def normalize_nationality(value: str | None) -> str | None:
    if not value:
        return None

    compact = re.sub(r"[^a-z]", "", _plain(value))
    if "vietnam" in compact or "vietnan" in compact:
        return "Việt Nam"
    return None


def normalize_gender(value: str | None) -> str | None:
    """
    Chuẩn hóa giới tính CCCD.

    Quy ước:
        Nam -> Nam
        Nữ  -> Nữ

    Chấp nhận các lỗi OCR thường gặp:
        Nu, Nữ, Ni, Nw, Nv, Nii -> Nu

    Không nhận "Nam" trong "Viet Nam".
    """

    if not value:
        return None

    text_value = str(value)

    # Một số file nguồn cũ đã giải mã UTF-8 bằng code page 437,
    # khiến "Nữ" trở thành "Nß╗»". Khôi phục chuỗi trước khi
    # chuẩn hóa để vẫn dùng được dữ liệu FIELD_OCR đã đọc đúng.
    try:
        repaired_value = text_value.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        repaired_value = text_value

    text = remove_accents(repaired_value)
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
        return "Nữ"

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
            line[:match.start()],
            line[match.end():],
        ]
        if index > 0:
            candidates.append(lines[index - 1])
        candidates.extend(lines[index + 1:index + 3])

        for value in candidates:
            candidate = normalize_full_name(value)
            if candidate:
                return candidate

    return None


def _reconcile_disputed_surname(
    selected_name: str,
    alternatives: list[str | None],
) -> str:
    """
    Chuẩn hóa họ chỉ khi nhiều nguồn cùng chữ gốc nhưng bất đồng dấu.

    Nhờ điều kiện bất đồng này, một kết quả đơn lẻ như ``HỔ`` sẽ không
    bị tự ý sửa. Với trường hợp raw OCR đọc ``HỔ`` và field OCR đọc
    ``HÔ``, họ phổ biến tương ứng được chuẩn hóa thành ``HỒ``.
    """
    selected_words = selected_name.split()
    if not selected_words:
        return selected_name

    surname_key = remove_accents(selected_words[0]).upper()
    canonical = CANONICAL_DISPUTED_SURNAMES.get(surname_key)
    if not canonical:
        return selected_name

    variants = {
        candidate.split()[0].upper()
        for candidate in alternatives
        if candidate
        and candidate.split()
        and remove_accents(candidate.split()[0]).upper()
        == surname_key
    }
    if len(variants) < 2:
        return selected_name

    selected_words[0] = canonical
    return " ".join(selected_words)

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


def _expiry_label_candidate(value: str) -> str:
    """Lấy phần có khả năng là nhãn hạn sử dụng, bỏ ngày và dữ liệu sau ngày."""
    original = str(value)
    normalized_digits = normalize_ocr_digits(original)
    date_match = re.search(
        r"(?<!\d)\d{1,2}[./-]\d{1,2}[./-]\d{4}(?!\d)",
        normalized_digits,
    )
    if date_match:
        original = original[:date_match.start()]

    plain = remove_accents(original).lower()
    plain = re.sub(r"[^a-z]+", " ", plain)
    return re.sub(r"\s+", " ", plain).strip()


def _looks_like_expiry_label(value: str) -> bool:
    """
    Nhận diện nhãn hạn sử dụng bị OCR sai nặng.

    Ví dụ thực tế:
        Duto of erpiry
        Dalo ar expiry
        Co gia ưn đen
        Co Igia E (triden

    Việc so khớp theo từng thành phần tránh nhầm ``Date of birth`` thành
    ``Date of expiry``.
    """
    plain_value = remove_accents(str(value))
    if re.search(
        EXPIRY_LABEL_PATTERN,
        plain_value,
        flags=re.IGNORECASE,
    ):
        return True

    candidate = _expiry_label_candidate(value)
    if not candidate:
        return False

    tokens = candidate.split()
    if len(tokens) >= 2:
        first_is_date = (
            SequenceMatcher(None, tokens[0], "date").ratio()
            >= 0.50
        )
        has_expiry = any(
            SequenceMatcher(None, token, "expiry").ratio()
            >= 0.55
            or token.endswith("piry")
            for token in tokens[1:]
        )
        if first_is_date and has_expiry:
            return True

    compact = "".join(tokens)
    has_gia = any(
        SequenceMatcher(None, token, "gia").ratio() >= 0.65
        or SequenceMatcher(None, token, "cogia").ratio() >= 0.70
        for token in tokens
    )
    last_token = tokens[-1] if tokens else ""
    has_den = bool(
        SequenceMatcher(None, last_token, "den").ratio() >= 0.62
        or last_token.endswith(("den", "oen"))
    )

    return bool(
        compact.startswith("c")
        and has_gia
        and has_den
        and SequenceMatcher(
            None,
            compact,
            "cogiatriden",
        ).ratio() >= 0.70
    )


def recover_labeled_date(
    raw_text: list[str] | None,
    field_name: str,
) -> str | None:
    """
    Phục hồi ngày từ đúng dòng chứa nhãn.

    Hỗ trợ dòng có đồng thời nhãn tiếng Việt và tiếng Anh, ví dụ:
        Ngay sinh / Date of birth: 24/0311995
    """

    if field_name == "dateOfExpiry":
        lines = [str(line) for line in raw_text or [] if line]

        for index, line in enumerate(lines):
            if not _looks_like_expiry_label(line):
                continue

            date = normalize_date(line)
            if date:
                return date

            # Nhãn và ngày đôi khi bị EasyOCR tách thành hai dòng.
            for next_line in lines[index + 1:index + 3]:
                date = normalize_date(next_line)
                if date:
                    return date

        return None

    patterns = {
        "dateOfBirth": (
            r"(?:ngay\s*sinh|date\s*of\s*birth)"
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

    text = str(value)

    # Tách theo thành phần địa chỉ để loại được cả nhãn hạn sử dụng bị
    # OCR sai nặng. Nếu cùng thành phần có ngày và địa chỉ ở bên phải,
    # chỉ bỏ nhãn + ngày và giữ phần địa chỉ.
    cleaned_pieces: list[str] = []
    for piece in re.split(r"[,;]", text):
        normalized_digits = normalize_ocr_digits(piece)
        date_match = re.search(
            r"(?<!\d)\d{1,2}[./-]\d{1,2}[./-]\d{4}(?!\d)",
            normalized_digits,
        )

        if date_match and _looks_like_expiry_label(
            piece[:date_match.start()]
        ):
            piece = piece[date_match.end():]
        elif _looks_like_expiry_label(piece):
            piece = ""

        if piece.strip(" :/-"):
            cleaned_pieces.append(piece)

    text = ", ".join(cleaned_pieces)

    def remove_pattern(pattern: str) -> None:
        """Xóa nhãn theo bản không dấu nhưng giữ dấu trong địa chỉ."""
        nonlocal text
        plain_text = remove_accents(text)
        matches = list(
            re.finditer(
                pattern,
                plain_text,
                flags=re.IGNORECASE,
            )
        )
        for match in reversed(matches):
            text = text[:match.start()] + " " + text[match.end():]

    # Loại cụm ngày hết hạn OCR sai nhẹ nhưng giữ phần địa chỉ phía sau.
    remove_pattern(
        rf"(?:{EXPIRY_LABEL_PATTERN})\s*"
        r"\d{1,2}[./-]\d{1,2}[./-]\d{4}"
    )

    remove_pattern(
        rf"(?:que\s*quan|{ORIGIN_EN_LABEL_PATTERN}|"
        rf"{RESIDENCE_VI_LABEL_PATTERN}|{RESIDENCE_EN_LABEL_PATTERN}|"
        rf"{EXPIRY_LABEL_PATTERN})\s*:?"
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
    return _looks_like_expiry_label(value)


def _contains_date(value: str) -> bool:
    return bool(
        re.search(
            r"(?<!\d)\d{1,2}[./-]\d{1,2}[./-]\d{4}(?!\d)",
            normalize_ocr_digits(value),
        )
    )


def _is_date_only_line(value: str) -> bool:
    normalized = normalize_ocr_digits(value)
    without_date = re.sub(
        r"(?<!\d)\d{1,2}[./-]\d{1,2}[./-]\d{4}(?!\d)",
        " ",
        normalized,
    )
    return _contains_date(value) and not re.search(
        r"[A-Za-zÀ-ỹ]",
        without_date,
    )


def _address_suffix_after_date(value: str) -> str:
    """Giữ phần địa chỉ ở bên phải ngày hết hạn trên dòng bị trộn cột."""
    matches = list(
        re.finditer(
            r"(?<!\d)\d{1,2}[./-]\d{1,2}[./-]\d{4}(?!\d)",
            normalize_ocr_digits(value),
        )
    )
    if not matches:
        return ""
    return value[matches[-1].end():].strip(" :;,/-\"")


def _looks_like_garbled_expiry_prefix(
    value: str,
    following_value: str | None,
) -> bool:
    """Nhận diện nửa nhãn hạn sử dụng bị OCR tách khỏi dòng ngày."""
    if not following_value or not _is_date_only_line(following_value):
        return False
    tokens = re.findall(r"[a-z0-9]+", _plain(value))
    return 1 <= len(tokens) <= 4 and "," not in value and ";" not in value


def _normalize_known_administrative_names(value: str) -> str:
    text = value

    for pattern, replacement in KNOWN_ADMINISTRATIVE_PHRASES:
        plain_text = remove_accents(text)
        matches = list(
            re.finditer(pattern, plain_text, flags=re.IGNORECASE)
        )
        for match in reversed(matches):
            text = text[:match.start()] + replacement + text[match.end():]

    return text


def _restore_address_accents(
    primary: str,
    alternatives: list[str | None],
) -> str:
    """
    Lấy lại dấu từ nguồn OCR khác nhưng giữ thứ tự của địa chỉ chính.

    Full-card OCR thường ghép dòng đúng hơn, còn field OCR được phóng to
    nên thường đọc dấu tốt hơn. Chỉ thay một từ khi các nguồn có dấu đều
    thống nhất về cùng một cách viết, tránh tự đoán dấu cho tên riêng.
    """
    def component_key(value: str) -> str:
        plain = remove_accents(value).casefold()
        plain = re.sub(r"[^a-z0-9]+", " ", plain)
        return re.sub(r"\s+", " ", plain).strip()

    alternative_components: dict[str, list[str]] = {}
    identifier_variants: dict[str, set[str]] = {}

    for alternative in alternatives:
        if not alternative:
            continue

        for component in re.split(r"[,;]", alternative):
            component = component.strip(" ,.;:/-")
            key = component_key(component)
            if key and component:
                alternative_components.setdefault(key, []).append(component)

        for token in re.findall(r"[^\W_]+", alternative, flags=re.UNICODE):
            if not any(character.isdigit() for character in token):
                continue
            key = remove_accents(token).casefold().translate(
                str.maketrans({"8": "b", "0": "o", "1": "i", "5": "s"})
            )
            identifier_variants.setdefault(key, set()).add(token)

    restored_components: list[str] = []
    for component in re.split(r"[,;]", primary):
        component = component.strip(" ,.;:/-")
        if not component:
            continue

        key = component_key(component)
        candidates = alternative_components.get(key, [])

        # Chỉ lấy nguyên cụm từ nguồn khác khi cụm chính hoàn toàn không
        # dấu và các nguồn có dấu thống nhất. So khớp theo cả cụm tránh
        # lỗi "Thành phố Thanh Hóa" biến thành "Thành phố Thành Hóa".
        if _diacritic_score(component) == 0 and candidates:
            accented = {
                candidate.casefold(): candidate
                for candidate in candidates
                if _diacritic_score(candidate) > 0
            }
            if len(accented) == 1:
                component = next(iter(accented.values()))

        restored_components.append(component)

    restored = ", ".join(restored_components)

    def replace_identifier(match: re.Match[str]) -> str:
        token = match.group(0)
        if not token.isdigit():
            return token
        key = token.casefold().translate(
            str.maketrans({"8": "b", "0": "o", "1": "i", "5": "s"})
        )
        mixed = [
            item
            for item in identifier_variants.get(key, set())
            if any(character.isdigit() for character in item)
            and any(character.isalpha() for character in item)
        ]
        return mixed[0] if len(mixed) == 1 else token

    restored = re.sub(r"\b\d+\b", replace_identifier, restored)
    return _normalize_known_administrative_names(restored)


def recover_address(
    raw_text: list[str] | None,
    field_name: str,
) -> str | None:
    lines = [str(line) for line in raw_text or [] if line]

    if field_name == "placeOfOrigin":
        label_pattern = (
            rf"(?:que\s*quan(?:\s*/?\s*(?:{ORIGIN_EN_LABEL_PATTERN}))?"
            rf"|{ORIGIN_EN_LABEL_PATTERN})\s*:?"
        )
        stop_pattern = (
            rf"(?:{RESIDENCE_VI_LABEL_PATTERN}|"
            rf"{RESIDENCE_EN_LABEL_PATTERN}|date\s*of\s*expiry)"
        )
        max_pieces = 2
    else:
        label_pattern = (
            rf"(?:{RESIDENCE_VI_LABEL_PATTERN}"
            rf"(?:\s*/?\s*(?:{RESIDENCE_EN_LABEL_PATTERN}))?"
            rf"|{RESIDENCE_EN_LABEL_PATTERN})\s*:?"
        )
        stop_pattern = r"\b(?:date\s*of\s*expiry)\b"
        max_pieces = 3

    for index, line in enumerate(lines):
        plain_line = remove_accents(line)
        label_match = re.search(
            label_pattern,
            plain_line,
            flags=re.IGNORECASE,
        )
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

        # Nếu line merger từng ghép nhầm hai cột, thứ tự vật lý là:
        # ngày hết hạn -> dòng cuối địa chỉ -> nhãn nơi thường trú ->
        # dòng đầu địa chỉ. Đưa phần sau nhãn lên trước rồi mới nối phần
        # địa chỉ nằm sau ngày để khôi phục đúng thứ tự.
        mixed_column_suffix = (
            _address_suffix_after_date(before)
            if field_name == "placeOfResidence"
            else ""
        )

        if mixed_column_suffix and usable(after):
            pieces.append(after)
            if usable(mixed_column_suffix):
                pieces.append(mixed_column_suffix)
        else:
            if usable(before):
                pieces.append(before)
            if usable(after):
                pieces.append(after)

        following_lines = lines[index + 1:index + 7]
        for offset, next_line in enumerate(following_lines):
            # EasyOCR sắp xếp theo tọa độ dọc. Trên CCCD, dòng
            # "Có giá trị đến" ở bên trái có thể nằm giữa hai dòng
            # nơi thường trú ở bên phải. Bỏ qua riêng dòng hạn sử dụng
            # và tiếp tục tìm phần địa chỉ còn lại trong cửa sổ hiện tại.
            if (
                field_name == "placeOfResidence"
                and _is_expiry_line(next_line)
            ):
                address_suffix = _address_suffix_after_date(
                    next_line
                )
                if usable(address_suffix):
                    pieces.append(address_suffix)
                continue

            following_value = (
                following_lines[offset + 1]
                if offset + 1 < len(following_lines)
                else None
            )
            if (
                field_name == "placeOfResidence"
                and _looks_like_garbled_expiry_prefix(
                    next_line,
                    following_value,
                )
            ):
                continue

            plain_next_line = remove_accents(next_line)
            if re.search(
                stop_pattern,
                plain_next_line,
                flags=re.IGNORECASE,
            ):
                break
            if STOP_LABEL_PATTERN.search(plain_next_line):
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

    candidates = (
        (raw_value, "RAW_TEXT_RECOVERY"),
        (full_value, "FULL_CARD_OCR"),
        (field_clean, "FIELD_OCR"),
    )

    for value, source in candidates:
        if value and is_valid_address(value, field_name):
            return (
                _restore_address_accents(
                    value,
                    [raw_value, full_value, field_clean],
                ),
                source,
            )

    return None, "NOT_FOUND"


def _diacritic_score(value: str) -> int:
    """Đếm dấu/ký tự riêng tiếng Việt để ưu tiên nguồn giàu dấu hơn."""
    decomposed = unicodedata.normalize("NFD", value)
    combining_marks = sum(
        unicodedata.category(character) == "Mn"
        for character in decomposed
    )
    vietnamese_d = sum(character in "Đđ" for character in value)
    return combining_marks + vietnamese_d


def _name_key(value: str) -> str:
    return re.sub(r"\s+", " ", remove_accents(value).upper()).strip()


def _restore_name_diacritics(
    value: str,
    gender_hint: str | None,
) -> str:
    """
    Phục hồi có kiểm soát cho họ phổ biến và tên đệm theo giới tính.

    Không dùng một từ điển chung để đoán mọi âm tiết vì cùng một chuỗi
    không dấu có thể tương ứng nhiều tên hợp lệ khác nhau.
    """
    if not gender_hint:
        return value

    words = value.split()
    if not words:
        return value

    surname_key = remove_accents(words[0]).upper()
    surname = COMMON_VIETNAMESE_SURNAMES.get(surname_key)
    if surname:
        words[0] = surname

    contextual_map: dict[str, str] = {}
    if gender_hint == "Nữ":
        contextual_map = FEMALE_NAME_DIACRITICS
    elif gender_hint == "Nam":
        contextual_map = MALE_NAME_DIACRITICS

    for index in range(1, len(words)):
        word_key = remove_accents(words[index]).upper()
        replacement = contextual_map.get(word_key)
        if replacement:
            words[index] = replacement

    return " ".join(words)


def select_full_name(
    field_value: str | None,
    full_card_value: str | None,
    raw_text: list[str] | None,
    gender_hint: str | None = None,
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

    full_name = normalize_full_name(full_card_value)
    raw_name = recover_full_name(raw_text)
    field_name = normalize_full_name(field_value)
    alternatives = [full_name, raw_name, field_name]
    candidates = [
        (full_name, "FULL_CARD_OCR"),
        (raw_name, "RAW_TEXT_RECOVERY"),
        (field_name, "FIELD_OCR"),
    ]

    selected_value: str | None = None
    selected_source = "NOT_FOUND"

    for candidate, source in candidates:
        if candidate:
            selected_value = candidate
            selected_source = source
            break

    if selected_value:
        # Nếu các nguồn cùng chữ gốc, chọn bản nhận được nhiều dấu hơn.
        equivalent_candidates = [
            (candidate, source)
            for candidate, source in candidates
            if candidate and _name_key(candidate) == _name_key(selected_value)
        ]
        selected_value, selected_source = max(
            equivalent_candidates,
            key=lambda item: _diacritic_score(item[0]),
        )
        selected_value = _reconcile_disputed_surname(
            selected_value,
            alternatives,
        )
        selected_value = _restore_name_diacritics(
            selected_value,
            gender_hint,
        )
        return selected_value, selected_source

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

    gender_hint = (
        recover_gender(raw_text)
        or normalize_gender(full_card.get("gender"))
        or normalize_gender(fields.get("gender"))
    )

    full_name, source = select_full_name(
        fields.get("fullName"),
        full_card.get("fullName"),
        raw_text,
        gender_hint=gender_hint,
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
