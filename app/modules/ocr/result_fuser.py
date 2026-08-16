from __future__ import annotations

import re
import unicodedata
from copy import deepcopy
from datetime import datetime
from difflib import SequenceMatcher
from statistics import median
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
    "TICH", "NATIONALITY",
}

ORIGIN_EN_LABEL_PATTERN = (
    r"place\s*[o0a][fl0]\s*o(?:ri|r|n|ni|i)?g(?:i)?n"
)
RESIDENCE_EN_LABEL_PATTERN = (
    r"place\s*[o0a][fl0]\s*['\"]?\s*"
    r"resi(?:d[eaou]n(?:c[eoa])?)?"
)
RESIDENCE_VI_LABEL_PATTERN = (
    r"n[o0](?:i|[1l])?\s*thu[o0]ng\s*tr?u"
)
BIRTH_LABEL_PATTERN = (
    r"(?:ngay(?:\s*[,;.]?\s*thang\s*[,;.]?\s*n[a-z]{1,3})?"
    r"\s*sinh|date\s*o[fl0]\s*birth|(?<!date\s)\bo[fl0]\s*birth)"
)
EXPIRY_LABEL_PATTERN = (
    r"(?:co|c[o0])\s*[i1l]?\s*[g9]ia(?:\s+[a-z])?\s*[\(\[]?\s*"
    r"(?:tr[iyj1l]|t[1il])\s*den|date\s*of\s*expiry"
)

STOP_LABEL_PATTERN = re.compile(
    rf"\b("
    r"ho\s*va{1,2}\s*ten|full\s*name|"
    rf"{BIRTH_LABEL_PATTERN}|"
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
    (r"\bthi\s+xa\s+phu\s+tho\b", "Thị xã Phú Thọ"),
    (r"\bthanh\s+pho\s+vinh\b", "Thành phố Vinh"),
    (r"\bthanh\s+pho\s+hai\s+duong\b", "Thành phố Hải Dương"),
    (r"\bthi\s+tran\s+voi\b", "Thị trấn Vôi"),
    (r"\bthi\s+tran\s+tam\s+binh\b", "Thị trấn Tam Bình"),
    (r"\bthi\s+xa\s+tu\s+son\b", "Thị xã Từ Sơn"),
    (r"\bthi\s*tran\s+thanh\s+lang\b", "Thị trấn Thanh Lãng"),
    (r"\btdp\s+dau\s+lang\b", "TDP Đầu Làng"),
    # Các mẫu dưới đây là lỗi ký tự có tính hệ thống của EasyOCR trên
    # chữ nhỏ/mờ. Chúng vẫn yêu cầu gần đủ cả cụm địa danh, không thay
    # một token đơn lẻ nên tránh biến chuỗi nhiễu thành địa chỉ giả.
    (r"\bt[nh][il1]\s+[sxw][aàá]\s+tu\s+son\b", "Thị xã Từ Sơn"),
    (r"\bin[l1]\s+w[aă]\s+tu\s+son\b", "Thị xã Từ Sơn"),
    (r"\b(?:chau|gra[su])\s*khe\b", "Châu Khê"),
    (r"\bthi(?:nh|nt|mt)\s+lang\b", "Thịnh Lang"),
    (r"\bdi(?:nh|nn|mn)\s+bang\b", "Đình Bảng"),
    (r"\bthi\s*x[a-z]?\s+tu\s+son\b", "Thị xã Từ Sơn"),
    (r"\btu\s+son\b", "Từ Sơn"),
    (r"\bthon\s+hoa\s+lieu\b", "Thôn Hòa Liễu"),
    (r"\bkv\s+thoi\s+thuan\b", "KV Thới Thuận"),
    (r"\bphu\s+van\s+nam\b", "Phú Vân Nam"),
    (r"\bthong\s+nhat\b", "Thống Nhất"),
    (r"\bthach\s+thang\b", "Thạch Thang"),
    (r"\bbach\s+thuan\b", "Bách Thuận"),
    (r"\bvu\s+thu\b", "Vũ Thư"),
    (r"\bthai\s+binh\b", "Thái Bình"),
    (r"\bhai\s+chau\b", "Hải Châu"),
    (r"\bhai\s+hau\b", "Hải Hậu"),
    (r"\bnam\s+dinh\b", "Nam Định"),
    (r"\bdoi\s+binh\b", "Đội Bình"),
    (r"\bung\s+(?:s[c0]{1,2}\s+)?hoa\b", "Ứng Hòa"),
    (r"\bha\s+noi\b", "Hà Nội"),
    (r"\bhung\s+viet\b", "Hùng Việt"),
    (r"\bcam\s+khe\b", "Cẩm Khê"),
    (r"\bphu\s+tho\b", "Phú Thọ"),
    (r"\bthanh\s+minh\b", "Thanh Minh"),
    (r"\bthuan\s+thien\b", "Thuận Thiên"),
    (r"\bkien\s+thuy\b", "Kiến Thụy"),
    (r"\bhai\s+phong\b", "Hải Phòng"),
    (r"\btien\s+thuy\b", "Tiến Thủy"),
    (r"\bquynh\s+luu\b", "Quỳnh Lưu"),
    (r"\bnghe\s+an\b", "Nghệ An"),
    (r"\bha\s+huy\s+tap\b", "Hà Huy Tập"),
    (r"\bmac\s+thi\s+buoi\b", "Mạc Thị Bưởi"),
    (r"\bhai\s+duong\b", "Hải Dương"),
    (r"\bthoi\s+an\s+dong\b", "Thới An Đông"),
    (r"\b[8b]inh\s+thuy\b", "Bình Thủy"),
    (r"\bcan\s+tho\b", "Cần Thơ"),
    (r"\bthoi\s+thuan\b", "Thới Thuận"),
    (r"\bphuoc\s+thoi\b", "Phước Thới"),
    (r"\bo\s+mon\b", "Ô Môn"),
    (r"\btuong\s+loc\b", "Tường Lộc"),
    (r"\btam\s+binh\b", "Tam Bình"),
    (r"\bvinh\s+lon[g]?\b", "Vĩnh Long"),
    (r"\btan\s+trung\b", "Tân Trung"),
    (r"\bmo\s+cay\s+nam\b", "Mỏ Cày Nam"),
    (r"\bphu\s+dinh\b", "Phú Định"),
    (r"\bphuong\s+16\b", "Phường 16"),
    (r"\bquan\s+8\b", "Quận 8"),
    (r"\btp\s+ho\s+chi\s+n\b", "TP Hồ Chí Minh"),
    (r"\btp\s+ho\s+chi\s+minh\b", "TP Hồ Chí Minh"),
    (r"\bthien\s+phien\b", "Thiện Phiến"),
    (r"\btien\s+lu\b", "Tiên Lữ"),
    (r"\bhung\s+yen\b", "Hưng Yên"),
    (r"\bbinh\s+xuyen\b", "Bình Xuyên"),
    (r"\bvinh\s+phuc\b", "Vĩnh Phúc"),
    (r"\bdong\s+son\b", "Đông Sơn"),
    (r"\bjogur[eê]an\b", "Nguyên"),
    (r"\b(?:[a-z]{0,3}guy[a-z]{1,8})\s+hong\b", "Nguyên Hồng"),
    (r"\bnguyen\s+hong\b", "Nguyên Hồng"),
    (r"\btan\s+son\b", "Tân Sơn"),
    (r"\bthanh\s+hoa\b", "Thanh Hóa"),
    (r"\bnam\s+binh\b", "Nam Bình"),
    (r"\b[sn]am\s+binh\b", "Nam Bình"),
    (r"\bk(?:i?e|ie)n\s+x(?:u|uw|ư)[a-z]{0,2}ng\b", "Kiến Xương"),
    (r"\bcao\s+mai\s+doa[i1l]\b", "Cao Mai Đoài"),
    (r"\bquang\s+trung\b", "Quang Trung"),
    (r"\bngach\b", "Ngách"),
    (r"\bnguyen\s+trai\b", "Nguyễn Trãi"),
    (r"\bnhan\s+chinh\b", "Nhân Chính"),
    (r"\bthanh[a-z]{0,5}c[aâ]n\b", "Thanh Xuân"),
    (r"\bthanh\s+xuan\b", "Thanh Xuân"),
    (r"\bea\s+hiao\b", "Ea Hiao"),
    (r"\bea\s+h\s*['’]?\s*leo\b", "Ea H'Leo"),
    (r"\bdak\s+lak\b", "Đắk Lắk"),
    (r"\bl[aàáeê]\s+(?:lo[i1]|ho[i1]n[gq])\b['’]?", "Lê Lợi"),
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


# Danh mục cấp tỉnh dùng để khôi phục dấu theo toàn cụm. Đây là dữ liệu
# miền bài toán (bao gồm tên cũ còn xuất hiện trên CCCD), không phải luật
# gắn với một ảnh cụ thể. Chỉ thay khi chuỗi không dấu khớp đúng tên tỉnh.
CANONICAL_PROVINCES: tuple[str, ...] = (
    "An Giang", "Bà Rịa - Vũng Tàu", "Bạc Liêu", "Bắc Giang",
    "Bắc Kạn", "Bắc Ninh", "Bến Tre", "Bình Dương", "Bình Định",
    "Bình Phước", "Bình Thuận", "Cà Mau", "Cao Bằng", "Cần Thơ",
    "Đà Nẵng", "Đắk Lắk", "Đắk Nông", "Điện Biên", "Đồng Nai",
    "Đồng Tháp", "Gia Lai", "Hà Giang", "Hà Nam", "Hà Nội",
    "Hà Tĩnh", "Hải Dương", "Hải Phòng", "Hậu Giang", "Hòa Bình",
    "Hưng Yên", "Khánh Hòa", "Kiên Giang", "Kon Tum", "Lai Châu",
    "Lạng Sơn", "Lào Cai", "Lâm Đồng", "Long An", "Nam Định",
    "Nghệ An", "Ninh Bình", "Ninh Thuận", "Phú Thọ", "Phú Yên",
    "Quảng Bình", "Quảng Nam", "Quảng Ngãi", "Quảng Ninh",
    "Quảng Trị", "Sóc Trăng", "Sơn La", "Tây Ninh", "Thái Bình",
    "Thái Nguyên", "Thanh Hóa", "Thừa Thiên Huế", "Tiền Giang",
    "Thành phố Hồ Chí Minh", "Trà Vinh", "Tuyên Quang", "Vĩnh Long",
    "Vĩnh Phúc", "Yên Bái", "Huế",
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

OCR_SURNAME_ALIASES = {
    # EasyOCR thường nhầm I hoa thành L ở họ BÙI.
    "BUL": "BÙI",
    "BU1": "BÙI",
}

FEMALE_NAME_DIACRITICS = {
    "THI": "THỊ",
    "MAY": "MÂY",
    "HONG": "HỒNG",
    "HUYEN": "HUYỀN",
    "DUYEN": "DUYÊN",
    "DIEU": "DIỆU",
}

MALE_NAME_DIACRITICS = {
    "VAN": "VĂN",
    "TUNG": "TÙNG",
    "QUOC": "QUỐC",
}

UNAMBIGUOUS_NAME_DIACRITICS = {
    "NGOC": "NGỌC",
    "THI": "THỊ",
    "QUOC": "QUỐC",
    "HIEP": "HIỆP",
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

    original = str(value)
    direct_match = re.search(r"(?<!\d)\d{12}(?!\d)", original)
    if direct_match:
        return direct_match.group(0)

    token_match = re.search(
        r"(?<![A-Za-z0-9])[0-9OoIl|]{12}(?![A-Za-z0-9])",
        original,
    )
    if token_match:
        digits = re.sub(
            r"\D",
            "",
            normalize_ocr_digits(token_match.group(0)),
        )
        if len(digits) == 12:
            return digits

    digits = re.sub(r"\D", "", normalize_ocr_digits(original))
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
    if re.search(
        r"vi[e3][tli1].{0,2}(?:nam|nan|[il1]?amn)",
        compact,
        flags=re.IGNORECASE,
    ):
        return "Việt Nam"

    # Ảnh mờ thường tách ``Việt Nam`` thành hai token như ``Vie Naii``.
    # Chỉ dùng fuzzy match cho một hoặc hai token bắt đầu bằng ``vi`` để
    # không biến một chuỗi OCR bất kỳ thành quốc tịch hợp lệ.
    tokens = re.findall(r"[a-z]+", _plain(value))
    fuzzy_candidates = list(tokens)
    fuzzy_candidates.extend(
        tokens[index] + tokens[index + 1]
        for index in range(len(tokens) - 1)
    )
    if any(
        candidate.startswith("vi")
        and 5 <= len(candidate) <= 9
        and SequenceMatcher(None, candidate, "vietnam").ratio() >= 0.70
        for candidate in fuzzy_candidates
    ):
        return "Việt Nam"

    return None


def collect_field_evidence(
    field_results: dict[str, Any] | None,
    field_name: str,
) -> list[str]:
    """Thu toàn bộ văn bản OCR của một field, không chỉ ứng viên thắng.

    Ảnh mờ thường tạo ra tình huống một biến thể đọc đúng ngày, biến thể
    khác đọc đúng dấu hoặc thứ tự địa chỉ. Trước đây pipeline chỉ chuyển
    ``structuredData`` vào fuser nên những bằng chứng đúng này bị mất.
    Hàm giữ từng biến thể độc lập để tầng hợp nhất có thể đồng thuận theo
    đúng kiểu dữ liệu của field.
    """
    if not isinstance(field_results, dict):
        return []

    result = field_results.get(field_name)
    if not isinstance(result, dict):
        return []

    candidates = result.get("ocrCandidates")
    if not isinstance(candidates, list):
        candidates = []

    evidence: list[str] = []

    def append_candidate(candidate: dict[str, Any]) -> None:
        local_values: list[str] = []
        for key in ("value", "joinedText"):
            value = candidate.get(key)
            if value is not None and str(value).strip():
                local_values.append(str(value).strip())

        for key in ("normalizedText", "rawText"):
            values = candidate.get(key)
            if not isinstance(values, list):
                continue
            local_values.extend(
                str(value).strip()
                for value in values
                if value is not None and str(value).strip()
            )

        # Không đếm lặp cùng một dòng trong một biến thể, nhưng vẫn giữ
        # sự đồng thuận khi hai biến thể ảnh độc lập cùng đọc một giá trị.
        seen: set[str] = set()
        for value in local_values:
            key = re.sub(r"\s+", " ", value).strip().casefold()
            if not key or key in seen:
                continue
            seen.add(key)
            evidence.append(value)

    for candidate in candidates:
        if isinstance(candidate, dict):
            append_candidate(candidate)

    if not evidence:
        append_candidate(result)

    return evidence


def recover_id_number(raw_text: list[str] | None) -> str | None:
    """Lấy số CCCD từ raw text khi parser/crop field bỏ sót."""
    lines = [str(line) for line in raw_text or [] if line]
    label_pattern = r"\b(?:so|s0|no)\b"

    for line in lines:
        plain_line = remove_accents(line)
        if not re.search(label_pattern, plain_line, flags=re.IGNORECASE):
            continue
        identifier = normalize_id_number(line)
        if identifier:
            return identifier

    # Một số lần EasyOCR tách riêng cụm 12 số khỏi nhãn Số / No.
    for line in lines:
        token_match = re.search(
            r"(?<![A-Za-z0-9])[0-9OoIl|]{12}(?![A-Za-z0-9])",
            line,
        )
        if token_match:
            identifier = normalize_id_number(token_match.group(0))
            if identifier:
                return identifier

    return None


def decode_id_demographics(
    identifier: str | None,
) -> tuple[int | None, str | None]:
    """Giải mã năm sinh và giới tính được mã hóa trong số CCCD.

    Chữ số thứ tư cho biết thế kỷ/giới tính; chữ số thứ năm và thứ sáu
    là hai số cuối của năm sinh. Dữ liệu này chỉ được dùng để sửa một
    ứng viên OCR ngày sinh đã có ngày/tháng, không tự tạo ngày sinh khi
    ảnh hoàn toàn không đọc được ngày.
    """
    normalized = normalize_id_number(identifier)
    if not normalized:
        return None, None

    code = int(normalized[3])
    century = 1900 + (code // 2) * 100
    birth_year = century + int(normalized[4:6])
    gender = "Nam" if code % 2 == 0 else "Nữ"
    return birth_year, gender


def _loose_date_parts(value: str | None) -> list[tuple[int, int, int]]:
    """Lấy ngày/tháng/năm kể cả khi OCR làm mất dấu phân cách thứ hai."""
    if not value:
        return []

    normalized = normalize_ocr_digits(str(value))
    matches = re.finditer(
        r"(?<!\d)(\d{1,2})\s*[/.-]\s*(\d{1,2})"
        r"\s*[/.-]?\s*(\d{4})(?!\d)",
        normalized,
    )

    parts: list[tuple[int, int, int]] = []
    for match in matches:
        day, month, year = (int(item) for item in match.groups())
        try:
            datetime(year=max(year, 1), month=month, day=day)
        except ValueError:
            continue
        parts.append((day, month, year))
    return parts


def reconcile_birth_date_with_id(
    identifier: str | None,
    selected_date: str | None,
    selected_source: str,
    field_value: str | None,
    full_card_value: str | None,
    raw_text: list[str] | None,
    extra_evidence: list[str] | None = None,
) -> tuple[str | None, str]:
    """Sửa riêng phần năm của ngày sinh bằng cấu trúc số CCCD.

    Ví dụ ảnh mờ đọc ``17/10/1801`` trong khi số CCCD chứa năm ``91``.
    Hàm giữ nguyên ngày/tháng nhìn thấy trên ảnh và chỉ thay năm khi chuỗi
    OCR khác năm mã hóa không quá hai chữ số. Nhờ vậy ngày hết hạn hoặc
    một chuỗi số ngẫu nhiên không bị biến thành ngày sinh.
    """
    expected_year, _ = decode_id_demographics(identifier)
    if expected_year is None:
        return selected_date, selected_source

    if selected_date:
        try:
            parsed = datetime.strptime(selected_date, "%d/%m/%Y")
        except ValueError:
            parsed = None
        if parsed is not None and parsed.year == expected_year:
            return selected_date, selected_source

    values: list[tuple[str, str, float]] = []
    if field_value:
        values.append((str(field_value), "FIELD_OCR", 3.0))
    if full_card_value:
        values.append((str(full_card_value), "FULL_CARD_OCR", 2.5))
    if selected_date:
        values.append((str(selected_date), selected_source, 2.0))
    values.extend(
        (str(line), "RAW_TEXT_RECOVERY", 1.0)
        for line in raw_text or []
        if line
    )
    values.extend(
        (str(value), "FIELD_OCR", 3.25)
        for value in extra_evidence or []
        if value
    )

    expected_text = f"{expected_year:04d}"
    candidates: list[tuple[float, str, str]] = []
    for value, source, base_score in values:
        plain = _plain(value)
        label_bonus = 2.0 if re.search(
            r"ngay\s*sinh|date\s*o[fl0]\s*birth|\bbirth\b|\bsinh\b",
            plain,
            flags=re.IGNORECASE,
        ) else 0.0

        for day, month, observed_year in _loose_date_parts(value):
            observed_text = f"{observed_year:04d}"
            digit_distance = sum(
                first != second
                for first, second in zip(observed_text, expected_text)
            )
            if digit_distance > 2:
                continue

            try:
                repaired = datetime(
                    year=expected_year,
                    month=month,
                    day=day,
                ).strftime("%d/%m/%Y")
            except ValueError:
                continue

            score = base_score + label_bonus + (2 - digit_distance) * 2.0
            candidates.append((score, repaired, source))

    if not candidates:
        return selected_date, selected_source

    _, repaired_date, evidence_source = max(
        candidates,
        key=lambda item: item[0],
    )
    if selected_date == repaired_date:
        return selected_date, selected_source
    return repaired_date, f"ID_STRUCTURE_RECOVERY({evidence_source})"


def _id_matches_birth_and_gender(
    identifier: str,
    date_of_birth: str | None,
    gender: str | None,
) -> tuple[bool | None, bool | None]:
    """Kiểm tra cấu trúc mã thế kỷ/giới tính/năm sinh của số CCCD."""
    if not re.fullmatch(r"\d{12}", identifier):
        return False, False

    birth_match: bool | None = None
    gender_match: bool | None = None
    parsed_birth: datetime | None = None
    if date_of_birth:
        try:
            parsed_birth = datetime.strptime(date_of_birth, "%d/%m/%Y")
        except ValueError:
            parsed_birth = None

    code = int(identifier[3])
    if parsed_birth is not None:
        birth_match = identifier[4:6] == f"{parsed_birth.year % 100:02d}"
        century_base = (parsed_birth.year // 100) - 19
        if century_base >= 0:
            expected_codes: set[int] = set()
            if gender == "Nam":
                expected_codes.add(century_base * 2)
            elif gender == "Nữ":
                expected_codes.add(century_base * 2 + 1)
            else:
                expected_codes.update(
                    {century_base * 2, century_base * 2 + 1}
                )
            gender_match = code in expected_codes
    elif gender:
        gender_match = (code % 2 == 0) if gender == "Nam" else (code % 2 == 1)

    return birth_match, gender_match


def select_id_number(
    full_card_value: str | None,
    field_value: str | None,
    raw_text: list[str] | None,
    date_of_birth: str | None,
    gender: str | None,
) -> tuple[str | None, str]:
    """Chọn số CCCD bằng đồng thuận nguồn và cấu trúc năm sinh/giới tính."""
    raw_value = recover_id_number(raw_text)
    candidates = [
        (normalize_id_number(full_card_value), "FULL_CARD_OCR", 2.4),
        (normalize_id_number(field_value), "FIELD_OCR", 2.0),
        (raw_value, "RAW_TEXT_RECOVERY", 2.2),
    ]
    available = [item for item in candidates if item[0]]
    if not available:
        return None, "NOT_FOUND"

    support = {
        identifier: sum(
            1 for candidate, _, _ in available if candidate == identifier
        )
        for identifier, _, _ in available
    }

    scored: list[tuple[float, str, str]] = []
    for identifier, source, source_score in available:
        birth_match, gender_match = _id_matches_birth_and_gender(
            identifier,
            date_of_birth,
            gender,
        )
        score = source_score + support[identifier] * 3.0
        if birth_match is True:
            score += 10.0
        elif birth_match is False:
            score -= 8.0
        if gender_match is True:
            score += 4.0
        elif gender_match is False:
            score -= 3.0
        scored.append((score, identifier, source))

    _, identifier, source = max(scored, key=lambda item: item[0])

    # Nếu nhiều nguồn cho cùng số, giữ provenance ưu tiên full-card rồi raw.
    agreeing_sources = [
        candidate_source
        for candidate, candidate_source, _ in candidates
        if candidate == identifier
    ]
    for preferred_source in (
        "FULL_CARD_OCR",
        "RAW_TEXT_RECOVERY",
        "FIELD_OCR",
    ):
        if preferred_source in agreeing_sources:
            source = preferred_source
            break

    return identifier, source


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
    label_pattern = r"(?:ho\s*va{1,2}\s*ten\s*/?\s*|full\s*name\s*:?)"

    for index, line in enumerate(lines):
        plain_line = remove_accents(line)
        match = re.search(label_pattern, plain_line, flags=re.IGNORECASE)
        if not match:
            continue

        candidates = [line[match.end():]]
        candidates.extend(lines[index + 1:index + 3])

        prefix = line[:match.start()]
        if not re.search(
            r"\bho\s*va{1,2}\s*te?n\b",
            remove_accents(prefix),
            flags=re.IGNORECASE,
        ):
            candidates.append(prefix)
        if index > 0:
            candidates.append(lines[index - 1])

        for value in candidates:
            candidate = normalize_full_name(value)
            if candidate:
                return candidate

    # Khi ảnh mờ, nhãn ``Họ và tên / Full name`` có thể hỏng hoàn toàn
    # nhưng dòng tên ngay sau số CCCD vẫn còn đọc được. Chỉ dùng fallback
    # này trong tối đa bốn dòng sau cụm 12 số và yêu cầu từ đầu là một họ
    # Việt Nam đã biết để không nhầm tiêu đề thẻ thành họ tên.
    identifier_indexes = [
        index
        for index, line in enumerate(lines)
        if normalize_id_number(line)
    ]
    for identifier_index in identifier_indexes:
        for value in lines[identifier_index + 1:identifier_index + 5]:
            if re.search(r"\d{3,}", normalize_ocr_digits(value)):
                continue

            # Khi nhãn tên bị OCR thành ``Ho va Jen / FW ratie``, nó
            # cũng nằm ngay sau số CCCD và có thể qua được kiểm tra họ
            # ``HO``. Bỏ riêng dòng có dấu gạch phân cột và ít nhất hai
            # token giống nhãn; dòng tên thật ngay kế tiếp vẫn được đọc.
            plain_tokens = re.findall(
                r"[A-Z]+",
                remove_accents(value).upper(),
            )
            if (
                "/" in value
                and plain_tokens
                and plain_tokens[0] == "HO"
            ):
                label_like_tokens = sum(
                    any(
                        SequenceMatcher(None, token, label).ratio() >= 0.60
                        for label in (
                            "VA", "TEN", "FULL", "NAME", "DATE", "BIRTH",
                        )
                    )
                    for token in plain_tokens[1:]
                )
                if label_like_tokens >= 2:
                    continue

            candidate = normalize_full_name(value)
            if not candidate:
                continue
            surname_key = remove_accents(candidate.split()[0]).upper()
            if (
                surname_key in COMMON_VIETNAMESE_SURNAMES
                or surname_key in OCR_SURNAME_ALIASES
            ):
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

    lines = [str(line) for line in raw_text or [] if line]
    for index, line in enumerate(lines):
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
            >= 0.30
            or token.endswith(("piry", "riny", "roiny"))
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

    label_similarity = SequenceMatcher(
        None,
        compact,
        "cogiatriden",
    ).ratio()

    return bool(
        compact.startswith("c")
        and has_gia
        and (
            (has_den and label_similarity >= 0.70)
            or label_similarity >= 0.78
        )
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
        "dateOfBirth": BIRTH_LABEL_PATTERN,
    }

    label_pattern = patterns.get(field_name)
    if not label_pattern:
        return None

    lines = [str(line) for line in raw_text or [] if line]
    for index, line in enumerate(lines):
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

        # Nhãn ngày sinh và giá trị có thể bị tách thành hai box/dòng.
        for next_line in lines[index + 1:index + 3]:
            date = normalize_date(next_line)
            if date:
                return date

    return None


def _clean_address_text(value: str | None) -> str | None:
    if not value:
        return None

    text = str(value)
    if "/:" in text:
        # Field crop đôi khi để lại phần cuối của nhãn song ngữ dưới dạng
        # "/:". Phần trước dấu này thuộc hàng phía trên, phần sau mới là
        # địa chỉ của field hiện tại.
        text = text.rsplit("/:", 1)[1]

    # Tách theo thành phần địa chỉ để loại được cả nhãn hạn sử dụng bị
    # OCR sai nặng. Nếu cùng thành phần có ngày và địa chỉ ở bên phải,
    # chỉ bỏ nhãn + ngày và giữ phần địa chỉ.
    cleaned_pieces: list[str] = []
    for piece in re.split(r"[,;]", text):
        # Dạng thực tế: ``... giá 49424103/2035 Nguyễn Trãi``. Cụm số
        # dài không thể là số nhà hợp lệ; bỏ phần nhiễu đến hết cụm đó
        # nhưng giữ địa chỉ nằm bên phải.
        long_date_noise = re.search(r"\d{5,}/\d{4}", piece)
        if long_date_noise and re.search(
            r"[A-Za-zÀ-ỹ]",
            piece[long_date_noise.end():],
        ):
            piece = piece[long_date_noise.end():]

        normalized_digits = normalize_ocr_digits(piece)
        date_match = re.search(
            r"(?<!\d)\d{1,2}[./-]\d{1,2}[./-]\d{4}(?!\d)",
            normalized_digits,
        )

        date_prefix = piece[:date_match.start()] if date_match else ""
        plain_date_prefix = _plain(date_prefix)
        prefix_compact = re.sub(r"[^a-z]", "", plain_date_prefix)
        if date_match and (
            _looks_like_expiry_label(date_prefix)
            or prefix_compact.startswith("coiga")
            or re.search(r"\btri\s*den\b", plain_date_prefix)
        ):
            piece = piece[date_match.end():]
        elif _looks_like_expiry_label(piece):
            piece = ""

        stripped_piece = piece.strip(" :/-\"'")
        plain_piece = _plain(stripped_piece)

        # Box OCR của hai cột thường tạo các thành phần rác chỉ gồm một
        # số hoặc một mẩu nhãn tiếng Anh. Không xóa số ở đầu địa chỉ vì
        # đó có thể là số nhà hợp lệ.
        if (
            cleaned_pieces
            and re.fullmatch(r"\d", stripped_piece)
        ):
            continue
        if (
            plain_piece.startswith("place ")
            and len(plain_piece.split()) <= 3
            and re.search(r"\b(?:origin|orunge|onigin|resid)", plain_piece)
        ):
            continue

        if stripped_piece:
            cleaned_pieces.append(stripped_piece)

    compact_pieces: list[str] = []
    for piece in cleaned_pieces:
        stripped = piece.strip(" ,.;:/-\"'")
        if not stripped:
            continue
        if (
            compact_pieces
            and re.fullmatch(r"\d{1,3}", stripped)
            and re.search(
                rf"(?<!\d){re.escape(stripped)}(?:/|\b)",
                compact_pieces[-1],
            )
        ):
            continue
        compact_pieces.append(stripped)

    if compact_pieces:
        compact_pieces[0] = re.sub(
            r"^(\d)\s+\1$",
            r"\1\1",
            compact_pieces[0],
        )

    if (
        len(compact_pieces) >= 2
        and re.fullmatch(r"\d{1,4}", compact_pieces[0])
        and re.search(r"[A-Za-zÀ-ỹ]", compact_pieces[1])
    ):
        compact_pieces[0:2] = [
            f"{compact_pieces[0]} {compact_pieces[1]}"
        ]

    text = ", ".join(compact_pieces)

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
    # Phần đầu của nhãn đôi khi bị rơi mất: "Of residenco".
    remove_pattern(
        r"\bo[fl0]\s*resi(?:d[eaou]n(?:c[eoa])?)?\s*:?"
    )
    # Chỉ xóa "Place of" còn dư ở đầu một thành phần; không xóa chữ
    # giữa dữ liệu để tránh sửa quá tay.
    text = re.sub(
        r"(^|,\s*)place\s*[o0][fl0]\s+",
        r"\1",
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
    # Số 0 đơn lẻ trước tên tỉnh thường là nhiễu OCR từ dấu/phần nền.
    text = re.sub(
        r"(^|,\s*)0\s+(?=[A-Za-zÀ-ỹ])",
        r"\1",
        text,
    )
    text = re.sub(
        r"(?<=[A-Za-zÀ-ỹ])\s+0\s+(?=[A-Za-zÀ-ỹ])",
        ", ",
        text,
    )
    text = re.sub(r"^(\d)\s+\1(?=,\s)", r"\1\1", text)
    text = re.sub(r"\b(\d)\s+\1\b(?=\s+[A-Za-zÀ-ỹ])", r"\1\1", text)
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
    if not following_value or not (
        _is_date_only_line(following_value)
        or _is_expiry_line(following_value)
        or _contains_date(following_value)
    ):
        return False
    tokens = re.findall(r"[a-z0-9]+", _plain(value))
    compact = "".join(tokens)
    looks_like_label_start = compact.startswith(("co", "c0", "da", "du"))
    return bool(
        1 <= len(tokens) <= 4
        and "," not in value.strip(" ,;:/-")
        and ";" not in value.strip(" ,;:/-")
        and looks_like_label_start
    )


def _restore_fuzzy_province_suffixes(value: str) -> str:
    """Sửa tên tỉnh ở cuối từng thành phần khi chỉ sai 1-2 ký tự OCR."""
    text = value
    replacements: list[tuple[int, int, str]] = []
    plain_text = remove_accents(text)

    component_ranges: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"[,;]", plain_text):
        component_ranges.append((start, match.start()))
        start = match.end()
    component_ranges.append((start, len(plain_text)))

    provinces = [
        (province, remove_accents(province).lower().split())
        for province in CANONICAL_PROVINCES
    ]
    # Chỉ thành phần cuối cùng là cấp tỉnh. Fuzzy từng thành phần sẽ dễ
    # biến ``Đông Sơn`` thành ``Đồng Nai`` hoặc ``Tam Bình`` thành một
    # tên tỉnh khác dù OCR ban đầu hoàn toàn đúng.
    for component_start, component_end in component_ranges[-1:]:
        component = plain_text[component_start:component_end]
        tokens = list(re.finditer(r"[A-Za-z]+", component))
        if not tokens:
            continue

        best: tuple[float, int, int, str] | None = None
        for province, province_words in provinces:
            word_count = len(province_words)
            if len(tokens) < word_count:
                continue
            selected = tokens[-word_count:]
            candidate = " ".join(item.group(0).lower() for item in selected)
            canonical = " ".join(province_words)
            if candidate[:1] != canonical[:1]:
                continue
            ratio = SequenceMatcher(None, candidate, canonical).ratio()
            threshold = 0.70 if word_count >= 2 else 0.82
            if ratio < threshold:
                continue
            absolute_start = component_start + selected[0].start()
            absolute_end = component_start + selected[-1].end()
            item = (ratio, absolute_start, absolute_end, province)
            if best is None or item[0] > best[0]:
                best = item

        if best is not None:
            _, match_start, match_end, province = best
            replacements.append((match_start, match_end, province))

    for match_start, match_end, province in reversed(replacements):
        text = text[:match_start] + province + text[match_end:]
    return text


def _looks_like_address_prefix_noise(value: str) -> bool:
    tokens = re.findall(r"[a-z]+", _plain(value))
    if not tokens:
        return False
    if any(len(token) >= 9 for token in tokens):
        return True
    if re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", _plain(value)):
        return True
    if (
        len(tokens) <= 5
        and any(len(token) == 1 for token in tokens)
        and not re.search(r"\d", value)
    ):
        return True
    label_words = ("place", "origin", "residence")
    return any(
        any(SequenceMatcher(None, token, label).ratio() >= 0.65 for label in label_words)
        for token in tokens
    )


def _trim_noise_before_known_component(value: str) -> str:
    """Bỏ prefix nhãn/rác nhưng chỉ khi sau nó có cụm địa danh đã biết."""
    text = value
    canonical_components = sorted(
        {
            replacement
            for _, replacement in KNOWN_ADMINISTRATIVE_PHRASES
            if len(replacement.split()) >= 2
        },
        key=len,
        reverse=True,
    )
    plain_text = remove_accents(text)
    matches: list[tuple[int, str]] = []
    for component in canonical_components:
        match = re.search(
            rf"\b{re.escape(remove_accents(component))}\b",
            plain_text,
            flags=re.IGNORECASE,
        )
        if match:
            matches.append((match.start(), component))
    if not matches:
        return text

    first_start, _ = min(matches, key=lambda item: item[0])
    prefix = text[:first_start].strip(" ,.;:/-'\"")
    if prefix and _looks_like_address_prefix_noise(prefix):
        return text[first_start:].lstrip(" ,.;:/-'\"")
    return text


def _insert_known_component_boundaries(value: str) -> str:
    """Khôi phục dấu phẩy giữa các cụm hành chính bị OCR ghép liền."""
    text = value
    boundary_phrases = (
        "Thị xã", "Thị trấn", "Thành phố", "Đình Bảng",
        "Kiến Xương", "Quang Trung", "Nguyễn Trãi", "Nhân Chính",
        "Thanh Xuân",
    )
    for phrase in boundary_phrases:
        plain_text = remove_accents(text)
        pattern = re.escape(remove_accents(phrase)).replace(r"\ ", r"\s+")
        matches = list(
            re.finditer(rf"(?<!^)\b{pattern}\b", plain_text, flags=re.IGNORECASE)
        )
        for match in reversed(matches):
            prefix = text[:match.start()].rstrip()
            if not prefix or prefix.endswith((",", ";", "/")):
                continue
            text = prefix + ", " + text[match.start():].lstrip()
    return text


def _repair_address_component_order(value: str) -> str:
    """Sửa trường hợp hai dòng địa chỉ bị ghép đảo do box OCR giao nhau."""
    components = [
        item.strip()
        for item in re.split(r"[,;]", value)
        if item.strip()
    ]
    if len(components) < 3:
        return value

    # Một box nhiễu có thể đứng trước dòng bắt đầu bằng số nhà. Nếu nó
    # không có chữ số/đơn vị hành chính và dòng kế tiếp có số nhà, bỏ
    # riêng box đầu thay vì giữ thành một thành phần địa chỉ.
    if (
        len(components) >= 3
        and not re.search(r"\d", components[0])
        and re.match(r"\d{1,4}\b", components[1])
        and not re.search(
            r"\b(?:thon|to|kp|khu|xa|phuong|huyen|quan|thi|thanh)\b",
            _plain(components[0]),
        )
    ):
        components.pop(0)

    numeric_component_index = next(
        (
            index
            for index, component in enumerate(components[:3])
            if re.fullmatch(r"\d{1,4}", component)
        ),
        None,
    )
    if numeric_component_index is not None and numeric_component_index > 0:
        noisy_prefix = components[:numeric_component_index]
        if any(
            re.search(r"[A-Za-zÀ-ỹ]\d|\d[A-Za-zÀ-ỹ]", item)
            or _looks_like_address_prefix_noise(item)
            for item in noisy_prefix
        ):
            components = components[numeric_component_index:]

    # EasyOCR có thể tách số nhà thành một component riêng trước tên
    # đường. Ghép lại trước khi kiểm tra quy tắc đảo dòng bên dưới.
    if (
        len(components) >= 2
        and re.fullmatch(r"\d{1,4}", components[0])
        and re.search(r"[A-Za-zÀ-ỹ]", components[1])
    ):
        components[0:2] = [f"{components[0]} {components[1]}"]

    first_match = re.fullmatch(r"(\d{1,4})\s+(.+)", components[0])
    if first_match:
        number, first_place = first_match.groups()
        first_key = _plain(first_place)
        province_keys = {
            _plain(province) for province in CANONICAL_PROVINCES
        }
        street_index = len(components) - 1
        if _plain(components[street_index]) in province_keys:
            street_index -= 1
        street_key = (
            _plain(components[street_index])
            if street_index > 0
            else ""
        )
        first_is_admin = first_key in {
            "tan son", "phuong", "xa", "thi tran", "thi xa",
        }
        last_is_street_name = bool(
            2 <= len(street_key.split()) <= 4
            and street_key not in province_keys
            and not street_key.startswith(("thanh pho", "thi xa", "thi tran"))
        )
        if first_is_admin and last_is_street_name:
            street = components[street_index]
            components = [
                f"{number} {street}",
                first_place,
                *components[1:street_index],
                *components[street_index + 1:],
            ]

    # Thành phố Thanh Hóa/Hải Dương vừa là đơn vị thành phố vừa thuộc
    # tỉnh cùng tên; khi OCR chỉ giữ một lần tên tỉnh, bổ sung thành phần
    # cấp tỉnh dựa trên chính cụm thành phố nhìn thấy.
    final_key = _plain(components[-1]) if components else ""
    for component in components:
        key = _plain(component)
        if not key.startswith("thanh pho "):
            continue
        province_key = key.removeprefix("thanh pho ").strip()
        province = next(
            (
                item
                for item in CANONICAL_PROVINCES
                if _plain(item) == province_key
            ),
            None,
        )
        if province and final_key != province_key:
            # OCR đôi khi tách riêng từ cuối của tên tỉnh, ví dụ
            # "Thành phố Thanh Hóa, Hóa". Bỏ mẩu lặp trước khi bổ sung
            # thành phần cấp tỉnh đầy đủ.
            if (
                len(final_key.split()) == 1
                and province_key.endswith(" " + final_key)
            ):
                components.pop()
            components.append(province)
            break

    return ", ".join(components)


def _normalize_known_administrative_names(value: str) -> str:
    text = value

    # Một số crop cắt cụt ``Hà Nội`` thành ``HàN``. Chỉ khôi phục trong
    # ngữ cảnh Thanh Xuân để không biến một địa danh ``Hà Nam`` hợp lệ.
    plain_context = remove_accents(text)
    context_matches = list(re.finditer(
        r"thanh[a-z]{0,5}c[aâ]n\s+ha\s*n[c0]?\b",
        plain_context,
        flags=re.IGNORECASE,
    ))
    for match in reversed(context_matches):
        text = (
            text[:match.start()]
            + "Thanh Xuân, Hà Nội"
            + text[match.end():]
        )

    for pattern, replacement in KNOWN_ADMINISTRATIVE_PHRASES:
        plain_text = remove_accents(text)
        matches = list(
            re.finditer(pattern, plain_text, flags=re.IGNORECASE)
        )
        for match in reversed(matches):
            text = text[:match.start()] + replacement + text[match.end():]

    # Chạy sau bước khôi phục ``Thanhgacân -> Thanh Xuân`` để xử lý cả
    # dấu phẩy/chấm phẩy ngăn giữa quận và tên tỉnh bị cắt cụt.
    plain_context = remove_accents(text)
    context_matches = list(re.finditer(
        r"thanh\s+xuan\s*[,;]?\s*ha\s*n[c0]?\b",
        plain_context,
        flags=re.IGNORECASE,
    ))
    for match in reversed(context_matches):
        text = (
            text[:match.start()]
            + "Thanh Xuân, Hà Nội"
            + text[match.end():]
        )

    text = _restore_fuzzy_province_suffixes(text)

    for province in sorted(CANONICAL_PROVINCES, key=len, reverse=True):
        province_key = remove_accents(province)
        pattern = re.escape(province_key).replace(r"\ ", r"\s+")
        plain_text = remove_accents(text)
        matches = list(
            re.finditer(
                rf"(?<![A-Za-z]){pattern}(?![A-Za-z])",
                plain_text,
                flags=re.IGNORECASE,
            )
        )
        for match in reversed(matches):
            text = text[:match.start()] + province + text[match.end():]

    # Chuẩn hóa từ chỉ đơn vị hành chính khi OCR nhận sai dấu thanh.
    administrative_terms = (
        (r"\bthi\s+xa\b", "Thị xã"),
        (r"\bthi\s+tran\b", "Thị trấn"),
        (r"\bthanh\s+pho\b", "Thành phố"),
        (r"\bphuong\b", "Phường"),
        (r"\bhuyen\b", "Huyện"),
        (r"\bthon\b", "Thôn"),
        (r"(?<!thi )\bxa\b", "Xã"),
    )
    for pattern, replacement in administrative_terms:
        plain_text = remove_accents(text)
        matches = list(
            re.finditer(pattern, plain_text, flags=re.IGNORECASE)
        )
        for match in reversed(matches):
            text = text[:match.start()] + replacement + text[match.end():]

    # OCR đôi khi làm mất dấu phẩy ngay trước tên tỉnh ở cuối địa chỉ.
    for province in sorted(CANONICAL_PROVINCES, key=len, reverse=True):
        plain_text = remove_accents(text)
        province_key = remove_accents(province)
        match = re.search(
            rf"(?<![A-Za-z]){re.escape(province_key)}\s*$",
            plain_text,
            flags=re.IGNORECASE,
        )
        if not match or match.start() == 0:
            continue
        prefix = text[:match.start()].rstrip()
        plain_prefix = remove_accents(prefix).lower()
        if prefix.endswith(",") or re.search(
            r"(?:thanh\s+pho|tinh)\s*$",
            plain_prefix,
        ):
            break
        text = prefix + ", " + text[match.start():].lstrip()
        break

    components = [
        component.strip()
        for component in text.split(",")
        if component.strip()
    ]
    if components and _plain(components[-1]) == "pho":
        city_marker_index = next(
            (
                index
                for index, component in enumerate(components[:-2])
                if _plain(component) == "thanh"
            ),
            None,
        )
        if city_marker_index is not None:
            city_name = components[city_marker_index + 1]
            components[city_marker_index:city_marker_index + 2] = [
                f"Thành phố {city_name}"
            ]
            components.pop()
            text = ", ".join(components)

    text = _trim_noise_before_known_component(text)
    text = _insert_known_component_boundaries(text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r",(?:\s*,)+", ", ", text)
    text = _repair_address_component_order(text)
    return text.strip(" ,;:/-")


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
    address_number_variants: dict[str, set[str]] = {}

    for alternative in alternatives:
        if not alternative:
            continue

        for component in re.split(r"[,;]", alternative):
            component = component.strip(" ,.;:/-'\"")
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

        for number in re.findall(r"\b\d{1,4}/\d{1,4}\b", alternative):
            key = re.sub(r"\D", "", number)
            address_number_variants.setdefault(key, set()).add(number)

    restored_components: list[str] = []
    canonical_province_keys = {
        component_key(province)
        for province in CANONICAL_PROVINCES
    }
    canonical_known_keys = {
        component_key(replacement)
        for _, replacement in KNOWN_ADMINISTRATIVE_PHRASES
    }
    for component in re.split(r"[,;]", primary):
        component = component.strip(" ,.;:/-'\"")
        if not component:
            continue

        key = component_key(component)
        candidates = alternative_components.get(key, [])
        if key:
            prefix_candidates = [
                candidate
                for candidate_key, values in alternative_components.items()
                if 0 < len(candidate_key) - len(key) <= 2
                and candidate_key.startswith(key)
                for candidate in values
            ]
            last_key_word = key.rsplit(" ", 1)[-1]
            if (
                len(last_key_word) <= 2
                and len({item.casefold() for item in prefix_candidates}) == 1
            ):
                candidates = prefix_candidates

            suffix_candidates = [
                candidate
                for candidate_key, values in alternative_components.items()
                if candidate_key.endswith(" " + key)
                and 0 < len(candidate_key) - len(key) <= 5
                and candidate_key in canonical_known_keys
                for candidate in values
            ]
            if (
                len(key.split()) == 1
                and len({
                    component_key(item) for item in suffix_candidates
                }) == 1
            ):
                candidates = suffix_candidates

        # Chỉ lấy nguyên cụm từ nguồn khác khi cụm chính hoàn toàn không
        # dấu và các nguồn có dấu thống nhất. So khớp theo cả cụm tránh
        # lỗi "Thành phố Thanh Hóa" biến thành "Thành phố Thành Hóa".
        is_prefix_completion = any(
            len(component_key(candidate)) > len(key)
            and (
                (
                    component_key(candidate).startswith(key)
                    and component_key(candidate) in canonical_province_keys
                )
                or (
                    component_key(candidate).endswith(" " + key)
                    and component_key(candidate) in canonical_known_keys
                )
            )
            for candidate in candidates
        )
        if candidates:
            accented = {
                candidate.casefold(): candidate
                for candidate in candidates
                if _diacritic_score(candidate) > 0
            }
            if accented:
                best_score = max(
                    _diacritic_score(candidate)
                    for candidate in accented.values()
                )
                best_candidates = {
                    candidate.casefold(): candidate
                    for candidate in accented.values()
                    if _diacritic_score(candidate) == best_score
                }
                if (
                    len(best_candidates) == 1
                    and (
                        _diacritic_score(component) < best_score
                        or is_prefix_completion
                    )
                ):
                    component = next(iter(best_candidates.values()))

        restored_components.append(component)

    restored = ", ".join(restored_components)

    def replace_identifier(match: re.Match[str]) -> str:
        token = match.group(0)
        if not token.isdigit():
            return token
        slash_variants = address_number_variants.get(token, set())
        if len(slash_variants) == 1:
            return next(iter(slash_variants))
        compatible_slash_variants = {
            number
            for variants in address_number_variants.values()
            for number in variants
            if token in {
                number.replace("/", ""),
                number.replace("/", "1"),
            }
        }
        if len(compatible_slash_variants) == 1:
            return next(iter(compatible_slash_variants))
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


def _restore_address_numbers(
    primary: str,
    alternatives: list[str | None],
) -> str:
    """Khôi phục dấu gạch chéo số nhà từ bằng chứng OCR lân cận.

    Crop lệch có thể đưa số nhà vào trường bên cạnh. Chỉ dùng trường đó
    để sửa một chuỗi số khi có đúng một dạng có dấu gạch chéo tương thích;
    tuyệt đối không dùng nội dung địa danh của trường bên cạnh.
    """
    slash_numbers = {
        number
        for alternative in alternatives
        if alternative
        for number in re.findall(r"\b\d{1,4}/\d{1,4}\b", alternative)
    }
    if not slash_numbers:
        return primary

    def replace_number(match: re.Match[str]) -> str:
        token = match.group(0)
        compatible = {
            number
            for number in slash_numbers
            if token in {
                number.replace("/", ""),
                number.replace("/", "1"),
            }
        }
        return next(iter(compatible)) if len(compatible) == 1 else token

    return re.sub(r"\b\d{3,9}\b", replace_number, primary)


def recover_address(
    raw_text: list[str] | None,
    field_name: str,
) -> str | None:
    lines = [str(line) for line in raw_text or [] if line]

    if field_name == "placeOfOrigin":
        label_pattern = (
            rf"(?:que\s*quan(?:\s*/?\s*(?:{ORIGIN_EN_LABEL_PATTERN}"
            rf"|place\s*[o0][fl0]\b))?"
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

        embedded_in_origin_line = bool(
            field_name == "placeOfResidence"
            and usable(before)
            and index > 0
            and re.search(
                rf"(?:que\s*quan|{ORIGIN_EN_LABEL_PATTERN})",
                remove_accents(lines[index - 1]),
                flags=re.IGNORECASE,
            )
        )

        if embedded_in_origin_line:
            # Dòng hiện tại vẫn là dữ liệu quê quán nhưng line merger đã
            # chen nhãn nơi thường trú vào giữa. Dữ liệu cư trú bắt đầu ở
            # dòng kế tiếp, vì vậy không đưa before/after vào kết quả.
            pass
        elif mixed_column_suffix and usable(after):
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
            if field_name == "placeOfOrigin" and re.search(
                rf"(?:{RESIDENCE_VI_LABEL_PATTERN}|"
                rf"{RESIDENCE_EN_LABEL_PATTERN})",
                plain_next_line,
                flags=re.IGNORECASE,
            ):
                # Hai cột có thể bị xen vào cùng một dòng theo thứ tự
                # ``Lê Lợi, Thành`` -> nhãn nơi thường trú ->
                # ``phố Bắc Giang, Bắc Giang``. Chỉ nối lại khi hai vế
                # khép đúng cụm ``Thành phố``; nhãn cư trú thông thường
                # không thỏa điều kiện hẹp này.
                residence_match = re.search(
                    RESIDENCE_EN_LABEL_PATTERN,
                    next_line,
                    flags=re.IGNORECASE,
                )
                if residence_match:
                    prefix = next_line[:residence_match.start()].strip()
                    suffix = next_line[residence_match.end():].strip(
                        " :;,/-\""
                    )
                    prefix = re.sub(
                        r"(?i)\btr[uú]\s*/?\s*$",
                        "",
                        prefix,
                    ).strip(" :;,/-\"")
                    city_word = re.search(
                        r"(?i)\bth[aàáảãạ]nh\s*$",
                        prefix,
                    )
                    if (
                        city_word
                        and _plain(suffix).startswith("pho ")
                    ):
                        reconstructed = _clean_address_text(
                            f"{prefix[:city_word.start()].strip()}, "
                            f"{prefix[city_word.start():].strip()} {suffix}"
                        )
                        if reconstructed and usable(reconstructed):
                            pieces.append(reconstructed)

                # Một số dòng bị trộn hai cột và chứa cả nhãn Việt lẫn
                # phần nhãn Anh bị cụt. Sau khi bỏ hai nhãn, phần còn lại
                # vẫn là quê quán của dòng đang đọc.
                if re.search(
                    r"(?<!place\s)\b[o0][fl0]\s*resi",
                    plain_next_line,
                    flags=re.IGNORECASE,
                ):
                    cleaned_mixed = _clean_address_text(next_line)
                    if cleaned_mixed and usable(cleaned_mixed):
                        pieces.append(cleaned_mixed)
                break
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
        if candidate:
            candidate = _normalize_known_administrative_names(candidate)
            candidate = _clean_address_text(candidate)
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
        is_partial_residence = bool(
            field_name == "placeOfResidence"
            and len(tokens) >= 2
            and re.search(
                r"^(?:thon|to|kp|khu|khoi|pho)\b",
                text,
            )
        )
        if not is_partial_residence:
            return False

    single_tokens = [token for token in tokens if len(token) == 1]
    if (
        len(single_tokens) >= 3
        and len(single_tokens) / max(len(tokens), 1) >= 0.25
    ):
        return False

    letter_tokens = re.findall(r"[a-z]+", text)
    if any(len(token) >= 8 for token in letter_tokens):
        return False
    if any(
        len(token) >= 4
        and not re.search(r"[aeiouy]", token)
        for token in letter_tokens
    ):
        return False
    if any(
        re.search(r"[a-z]", token)
        and re.search(r"\d", token)
        and not re.fullmatch(r"\d{1,4}[a-z]", token)
        for token in tokens
    ):
        return False

    forbidden = (
        "full name", "date of birth", "gioi tinh", "sex",
        "quoc tich", "nationality", "date of expiry",
        "co gia tri den", "gia tri den",
        "place of origin", "place of residence",
        "of origin", "of residence",
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

    label_tokens = {
        "que", "quan", "place", "origin", "residence",
        "nationality", "thuong", "tru", "expiry",
    }
    if sum(token in label_tokens for token in tokens) >= 2:
        return False
    if re.search(r"(?<!\d)/(?!\d)", str(value)):
        return False

    return True


def _canonical_spatial_boxes(
    text_boxes: list[dict[str, Any]] | None,
    image_size: tuple[int, int] | list[int] | None = None,
) -> list[dict[str, Any]]:
    """Chuẩn hóa box OCR về hệ tọa độ thẻ 1000 x 630.

    OCR toàn thẻ giữ được tọa độ vật lý ngay cả khi line merger ghép sai
    thứ tự hai cột ở đáy thẻ. Tầng hợp nhất dùng các box nguyên thủy này
    để dựng lại riêng quê quán và nơi thường trú.
    """
    normalized: list[dict[str, Any]] = []
    raw_boxes: list[tuple[dict[str, Any], float, float, float, float]] = []

    for item in text_boxes or []:
        if not isinstance(item, dict):
            continue
        points = item.get("box")
        text = normalize_spaces(str(item.get("text") or ""))
        if not text or not isinstance(points, list) or len(points) < 4:
            continue
        try:
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
        except (TypeError, ValueError, IndexError):
            continue
        if not xs or not ys:
            continue
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        if right <= left or bottom <= top:
            continue
        raw_boxes.append((item, left, top, right, bottom))

    if not raw_boxes:
        return []

    width = 0.0
    height = 0.0
    if (
        isinstance(image_size, (tuple, list))
        and len(image_size) >= 2
    ):
        try:
            width = float(image_size[0])
            height = float(image_size[1])
        except (TypeError, ValueError):
            width = height = 0.0

    if width <= 0 or height <= 0:
        # Mặt thẻ đã warp theo tỉ lệ 1000:630. Chừa một lề nhỏ vì box
        # ngoài cùng hiếm khi chạm đúng biên ảnh.
        maximum_right = max(item[3] for item in raw_boxes)
        width = maximum_right / 0.965
        height = width * 0.63

    scale_x = 1000.0 / max(width, 1.0)
    scale_y = 630.0 / max(height, 1.0)
    for item, left, top, right, bottom in raw_boxes:
        normalized.append({
            "text": str(item.get("text") or "").strip(),
            "confidence": float(item.get("confidence") or 0.0),
            "_left": left * scale_x,
            "_top": top * scale_y,
            "_right": right * scale_x,
            "_bottom": bottom * scale_y,
            "_center_x": (left + right) * 0.5 * scale_x,
            "_center_y": (top + bottom) * 0.5 * scale_y,
            "_height": max((bottom - top) * scale_y, 1.0),
        })

    return normalized


def estimate_layout_y_offset(
    text_boxes: list[dict[str, Any]] | None,
    image_size: tuple[int, int] | list[int] | None = None,
) -> float:
    """Ước lượng độ lệch dọc của mẫu in từ vị trí cụm 12 số CCCD."""
    boxes = _canonical_spatial_boxes(text_boxes, image_size)
    identifier_centers: list[float] = []
    identifier_pattern = re.compile(
        r"(?<![A-Za-z0-9])([0-9OoIl|]{12})(?![A-Za-z0-9])"
    )
    for item in boxes:
        matches = identifier_pattern.finditer(str(item["text"]))
        if any(
            sum(character.isdigit() for character in match.group(1)) >= 8
            for match in matches
        ):
            identifier_centers.append(item["_center_y"])
    if not identifier_centers:
        return 0.0

    # Trên mẫu chuẩn, tâm cụm số nằm quanh y=285. Các ảnh test mới lệch
    # cả khối chữ khoảng +/-42 px; clamp ngăn một box nhận sai kéo toàn
    # bộ crop ra ngoài thẻ.
    offset = float(median(identifier_centers)) - 285.0
    return round(max(-55.0, min(55.0, offset)), 2)


def _spatial_label_kinds(value: str) -> set[str]:
    """Nhận diện các nhãn field có thể cùng nằm trong một OCR box."""
    plain = _plain(value)
    compact = re.sub(r"[^a-z]", "", plain)
    tokens = re.findall(r"[a-z]+", plain)
    kinds: set[str] = set()

    def fuzzy_token(target: str, threshold: float = 0.67) -> bool:
        return any(
            SequenceMatcher(None, token, target).ratio() >= threshold
            for token in tokens
        )

    if (
        re.search(r"\b(?:s[o06]|no)\s*(?:/|:|\b)", plain)
        or "personalidentificationnumber" in compact
    ):
        kinds.add("idNumber")

    if (
        re.search(r"\bho\b.{0,28}\bten\b", plain)
        or re.search(r"\bfull\s*n?a?m?e?\b", plain)
        or (fuzzy_token("full", 0.72) and fuzzy_token("name", 0.60))
    ):
        kinds.add("fullName")

    if (
        re.search(r"\bngay\b.{0,26}\bsinh\b", plain)
        or re.search(r"\bdate\b.{0,20}\bb(?:ir|ut|irt)[a-z]*\b", plain)
        or ("sinh" in tokens and "ngay" in tokens)
    ):
        kinds.add("dateOfBirth")

    if (
        re.search(r"\bgioi\s*t[i1l]nh\b|\bsex\b", plain)
        or (fuzzy_token("gioi", 0.70) and fuzzy_token("tinh", 0.65))
    ):
        kinds.add("gender")

    if (
        re.search(r"\bqu[o0]c\s*t[i1l]ch\b|\bnational", plain)
        or (fuzzy_token("quoc", 0.70) and fuzzy_token("tich", 0.65))
    ):
        kinds.add("nationality")

    if (
        re.search(r"\bque\s*quan\b", plain)
        or (
            "place" in tokens
            and any(
                SequenceMatcher(None, token, "origin").ratio() >= 0.62
                for token in tokens
            )
        )
        or re.search(r"\bo[fl0]\s*origin\b", plain)
    ):
        kinds.add("placeOfOrigin")

    fuzzy_vi_residence = False
    for index, token in enumerate(tokens):
        if token != "noi":
            continue
        tail = tokens[index + 1:index + 5]
        for local_index, candidate in enumerate(tail):
            if (
                SequenceMatcher(
                    None,
                    candidate,
                    "thuong",
                ).ratio()
                < 0.68
            ):
                continue
            if any(
                SequenceMatcher(None, following, "tru").ratio() >= 0.60
                for following in tail[local_index + 1:local_index + 3]
            ):
                fuzzy_vi_residence = True
                break
        if fuzzy_vi_residence:
            break

    if (
        re.search(r"\bnoi\s*thuong\s*tru\b", plain)
        or fuzzy_vi_residence
        or (
            "place" in tokens
            and any(
                token.startswith("res")
                or SequenceMatcher(None, token, "residence").ratio() >= 0.45
                for token in tokens
            )
        )
        or any(token.startswith("residence") for token in tokens)
        or (
            compact.startswith("resid")
            and len(compact) >= 6
        )
    ):
        kinds.add("placeOfResidence")

    if (
        re.search(r"\bco\b.{0,24}\bgia\b.{0,16}\btri\b.{0,12}\bden\b", plain)
        or re.search(r"\bdate\b.{0,16}\bexpiry\b|\bexpiry\s*date\b", plain)
    ):
        kinds.add("dateOfExpiry")

    return kinds


def _spatial_label_kind(value: str) -> str | None:
    """Giữ API cũ dành riêng cho hai nhãn địa chỉ."""
    kinds = _spatial_label_kinds(value)
    for field_name in ("placeOfOrigin", "placeOfResidence"):
        if field_name in kinds:
            return field_name

    return None


def _same_spatial_row(
    first: dict[str, Any],
    second: dict[str, Any],
) -> bool:
    overlap = max(
        0.0,
        min(first["_bottom"], second["_bottom"])
        - max(first["_top"], second["_top"]),
    )
    smaller_height = max(
        min(first["_height"], second["_height"]),
        1.0,
    )
    center_distance = abs(
        first["_center_y"] - second["_center_y"]
    )
    return bool(
        overlap / smaller_height >= 0.35
        or center_distance <= min(
            max(first["_height"], second["_height"]) * 0.42,
            24.0,
        )
    )


def estimate_field_crop_layout(
    text_boxes: list[dict[str, Any]] | None,
    image_size: tuple[int, int] | list[int] | None = None,
) -> dict[str, Any]:
    """Dựng vùng cắt từng hàng từ vị trí nhãn thật trên CCCD.

    Tọa độ mẫu chỉ đóng vai trò giới hạn an toàn. Khi nhận ra nhãn, mép
    trên của nhãn hiện tại và nhãn kế tiếp tạo thành một dải không chồng
    lấn, nhờ vậy họ tên không ăn sang ngày sinh và quê quán không ăn sang
    nơi thường trú.
    """
    boxes = _canonical_spatial_boxes(text_boxes, image_size)
    offset = estimate_layout_y_offset(text_boxes, image_size)
    expected_centers = {
        "idNumber": 285.0 + offset,
        "fullName": 335.0 + offset,
        "dateOfBirth": 405.0 + offset,
        "gender": 445.0 + offset,
        "nationality": 445.0 + offset,
        "placeOfOrigin": 490.0 + offset,
        "placeOfResidence": 550.0 + offset,
        "dateOfExpiry": 565.0 + offset,
    }

    anchors: dict[str, dict[str, float]] = {}
    for field_name, expected_center in expected_centers.items():
        candidates = [
            item
            for item in boxes
            if field_name in _spatial_label_kinds(str(item["text"]))
            and abs(float(item["_center_y"]) - expected_center) <= 92.0
        ]
        if not candidates:
            continue
        anchor = min(
            candidates,
            key=lambda item: abs(float(item["_center_y"]) - expected_center),
        )
        same_row = [
            item for item in candidates if _same_spatial_row(anchor, item)
        ]
        anchors[field_name] = {
            "left": round(min(float(item["_left"]) for item in same_row), 2),
            "top": round(min(float(item["_top"]) for item in same_row), 2),
            "right": round(max(float(item["_right"]) for item in same_row), 2),
            "bottom": round(max(float(item["_bottom"]) for item in same_row), 2),
            "center": round(float(median(
                float(item["_center_y"]) for item in same_row
            )), 2),
        }

    field_templates = {
        "idNumber": (255, 230, 930, 330),
        "fullName": (270, 315, 975, 395),
        "dateOfBirth": (270, 380, 900, 450),
        "gender": (270, 405, 650, 480),
        "nationality": (620, 405, 995, 480),
        "placeOfOrigin": (300, 455, 995, 530),
        "placeOfResidence": (300, 530, 995, 630),
        "dateOfExpiry": (0, 515, 345, 630),
    }
    value_templates = {
        "idNumber": (330, 245, 850, 325),
        "fullName": (270, 325, 985, 400),
        "dateOfBirth": (500, 380, 860, 455),
        "gender": (400, 405, 625, 480),
        "nationality": (730, 405, 1000, 480),
        "placeOfOrigin": (300, 455, 1000, 530),
        "placeOfResidence": (300, 530, 1000, 630),
        "dateOfExpiry": (15, 520, 345, 630),
    }

    next_anchor = {
        "idNumber": "fullName",
        "fullName": "dateOfBirth",
        "dateOfBirth": "gender",
        "gender": "placeOfOrigin",
        "nationality": "placeOfOrigin",
        "placeOfOrigin": "placeOfResidence",
    }

    def resolve_template(
        field_name: str,
        template: tuple[int, int, int, int],
    ) -> dict[str, int]:
        x1, base_y1, x2, base_y2 = template
        shifted_y1 = int(round(base_y1 + offset))
        shifted_y2 = (
            630
            if field_name in {"placeOfResidence", "dateOfExpiry"}
            else int(round(base_y2 + offset))
        )
        y1 = shifted_y1
        y2 = shifted_y2
        anchor = anchors.get(field_name)
        if anchor:
            proposed_y1 = int(round(float(anchor["top"]) - 7.0))
            y1 = max(
                shifted_y1 - 30,
                min(shifted_y1 + 30, proposed_y1),
            )

        following = anchors.get(next_anchor.get(field_name, ""))
        if following:
            proposed_y2 = int(round(float(following["top"]) - 3.0))
            y2 = max(
                shifted_y2 - 35,
                min(shifted_y2 + 35, proposed_y2),
            )

        y1 = max(0, min(628, y1))
        y2 = max(y1 + 24, min(630, y2))
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    regions: dict[str, dict[str, dict[str, int]]] = {}
    for field_name, template in field_templates.items():
        regions[field_name] = {
            "field": resolve_template(field_name, template),
            "value": resolve_template(
                field_name,
                value_templates[field_name],
            ),
        }

    # Mỗi cặp hàng dùng đúng một ranh giới chung. Không để dải họ tên ăn
    # xuống ngày sinh hoặc ngày sinh ăn xuống giới tính như các template
    # rộng trước đây. Mép nhãn hàng dưới là ưu tiên, midpoint template là
    # fallback khi OCR toàn thẻ chưa thấy nhãn.
    adjacent_rows = (
        ("idNumber", "fullName"),
        ("fullName", "dateOfBirth"),
        ("dateOfBirth", "gender"),
        ("gender", "placeOfOrigin"),
    )
    for upper_field, lower_field in adjacent_rows:
        lower_anchor = anchors.get(lower_field)
        if lower_anchor:
            boundary = int(round(float(lower_anchor["top"]) - 3.0))
        else:
            boundary = int(round((
                regions[upper_field]["field"]["y2"]
                + regions[lower_field]["field"]["y1"]
            ) * 0.5))
        upper_start = regions[upper_field]["field"]["y1"]
        lower_end = regions[lower_field]["field"]["y2"]
        boundary = max(upper_start + 24, min(lower_end - 24, boundary))
        for kind in ("field", "value"):
            regions[upper_field][kind]["y2"] = boundary
            regions[lower_field][kind]["y1"] = boundary

    # Hai cột giới tính và quốc tịch luôn cùng một hàng, kể cả khi EasyOCR
    # tách thành hai box hoặc chỉ nhận ra một trong hai nhãn.
    shared_start = int(regions["gender"]["field"]["y1"])
    shared_end = int(regions["gender"]["field"]["y2"])
    for field_name in ("gender", "nationality"):
        for kind in ("field", "value"):
            regions[field_name][kind]["y1"] = shared_start
            regions[field_name][kind]["y2"] = shared_end

    default_boundary = 530.0 + offset
    boundary = default_boundary
    residence_anchor = anchors.get("placeOfResidence")
    if residence_anchor:
        boundary = max(
            default_boundary - 45.0,
            min(default_boundary + 60.0, float(residence_anchor["top"]) - 2.0),
        )
    origin_anchor = anchors.get("placeOfOrigin")
    if origin_anchor:
        boundary = max(boundary, float(origin_anchor["bottom"]) + 6.0)
    boundary_y = int(round(max(470.0, min(585.0, boundary))))
    for kind in ("field", "value"):
        regions["placeOfOrigin"][kind]["y2"] = boundary_y
        regions["placeOfResidence"][kind]["y1"] = boundary_y
        regions["placeOfResidence"][kind]["y2"] = 630

    return {
        "source": "label_anchors" if anchors else "template_with_layout_offset",
        "layoutYOffset": offset,
        "boundaryY": boundary_y,
        "labelAnchors": anchors,
        "regions": regions,
    }


def estimate_address_crop_layout(
    text_boxes: list[dict[str, Any]] | None,
    image_size: tuple[int, int] | list[int] | None = None,
) -> dict[str, Any]:
    """Định vị ranh giới hai vùng địa chỉ từ nhãn trên chính thẻ.

    ``layoutYOffset`` vẫn là fallback cho ảnh mà OCR chưa nhận ra nhãn.
    Khi thấy nhãn ``Nơi thường trú / Place of residence``, mép trên của
    nhãn trở thành ranh giới dùng chung: crop quê quán kết thúc tại đó và
    crop nơi thường trú bắt đầu tại đó. Vì vậy hai crop không thể chồng lấn.
    """
    boxes = _canonical_spatial_boxes(text_boxes, image_size)
    offset = estimate_layout_y_offset(text_boxes, image_size)
    default_boundary = 530.0 + offset
    expected_centers = {
        "placeOfOrigin": 490.0 + offset,
        "placeOfResidence": 550.0 + offset,
    }
    anchors: dict[str, dict[str, float]] = {}

    for field_name, expected_center in expected_centers.items():
        candidates = [
            item
            for item in boxes
            if _spatial_label_kind(item["text"]) == field_name
            and abs(item["_center_y"] - expected_center) <= 82.0
        ]
        if not candidates:
            continue
        anchor = min(
            candidates,
            key=lambda item: abs(item["_center_y"] - expected_center),
        )
        same_row = [
            item
            for item in candidates
            if _same_spatial_row(anchor, item)
        ]
        anchors[field_name] = {
            "top": round(min(item["_top"] for item in same_row), 2),
            "bottom": round(max(item["_bottom"] for item in same_row), 2),
            "center": round(float(median(
                item["_center_y"] for item in same_row
            )), 2),
        }

    boundary = default_boundary
    source = "template_with_layout_offset"
    residence_anchor = anchors.get("placeOfResidence")
    if residence_anchor:
        anchored_boundary = float(residence_anchor["top"]) - 2.0
        # Chỉ cho nhãn dịch ranh giới trong một khoảng an toàn. Một box OCR
        # nhận nhầm ở đầu hoặc cuối thẻ không được kéo crop sang trường khác.
        boundary = max(
            default_boundary - 45.0,
            min(default_boundary + 60.0, anchored_boundary),
        )
        source = "residence_label"

    origin_anchor = anchors.get("placeOfOrigin")
    if origin_anchor:
        boundary = max(
            boundary,
            float(origin_anchor["bottom"]) + 6.0,
        )

    boundary = round(max(470.0, min(585.0, boundary)), 2)
    return {
        "boundaryY": boundary,
        "source": source,
        "layoutYOffset": offset,
        "labelAnchors": anchors,
    }


def _join_spatial_row(
    boxes: list[dict[str, Any]],
) -> str:
    ordered = sorted(boxes, key=lambda item: item["_left"])
    if len(ordered) > 1:
        ordered = [
            item
            for item in ordered
            if str(item["text"]).strip() not in {"0", "1"}
        ]
    return normalize_spaces(
        " ".join(str(item["text"]) for item in ordered)
    ) or ""


def recover_spatial_address(
    text_boxes: list[dict[str, Any]] | None,
    field_name: str,
    image_size: tuple[int, int] | list[int] | None = None,
) -> str | None:
    """Dựng địa chỉ từ tọa độ box, không dùng thứ tự line merger.

    Dòng hạn sử dụng nằm bên trái và dòng địa chỉ nằm bên phải. Khi hai
    dòng có y gần nhau, EasyOCR có thể trả thứ tự đảo hoặc trộn ngày vào
    địa chỉ. Cách dựng theo cột giải quyết trường hợp đó trước khi parser
    làm sạch văn bản.
    """
    if field_name not in {"placeOfOrigin", "placeOfResidence"}:
        return None

    boxes = _canonical_spatial_boxes(text_boxes, image_size)
    if not boxes:
        return None

    offset = estimate_layout_y_offset(text_boxes, image_size)
    expected_label_y = (
        490.0 + offset
        if field_name == "placeOfOrigin"
        else 550.0 + offset
    )
    label_boxes = [
        item
        for item in boxes
        if _spatial_label_kind(item["text"]) == field_name
    ]
    if not label_boxes:
        return None

    anchor = min(
        label_boxes,
        key=lambda item: abs(item["_center_y"] - expected_label_y),
    )
    anchor_row_labels = [
        item
        for item in label_boxes
        if _same_spatial_row(anchor, item)
    ]
    anchor_left = min(item["_left"] for item in anchor_row_labels)
    anchor_center = float(median(
        item["_center_y"] for item in anchor_row_labels
    ))
    anchor_height = max(float(median(
        item["_height"] for item in anchor_row_labels
    )), 1.0)

    def is_other_label(item: dict[str, Any]) -> bool:
        kind = _spatial_label_kind(item["text"])
        return bool(kind and kind != field_name)

    def is_expiry_column(item: dict[str, Any]) -> bool:
        plain = _plain(item["text"])
        return bool(
            item["_right"] <= 335.0
            and (
                _looks_like_expiry_label(item["text"])
                or _contains_date(item["text"])
                or re.search(r"\b(?:gia|tri|den|expiry)\b", plain)
            )
        )

    def is_label_fragment(item: dict[str, Any]) -> bool:
        plain = re.sub(r"[^a-z]", "", _plain(item["text"]))
        return plain in {
            "que", "quan", "noi", "thuong", "tru",
            "place", "of", "origin", "residence",
        }

    if field_name == "placeOfOrigin":
        residence_labels = [
            item
            for item in boxes
            if _spatial_label_kind(item["text"]) == "placeOfResidence"
            and item["_center_y"] > anchor_center
        ]
        upper_y = anchor_center + 92.0
        if residence_labels:
            residence_center = min(
                item["_center_y"] for item in residence_labels
            )
            # Box quê quán cao có thể chạm nhẹ hàng nhãn cư trú, nên
            # chừa 4 px nhưng không lấy giá trị cư trú ở bên phải.
            upper_y = min(upper_y, residence_center + 4.0)
        candidates = [
            item
            for item in boxes
            if item["_center_x"] >= 265.0
            and anchor_center + 7.0
            <= item["_center_y"]
            <= upper_y
            and _spatial_label_kind(item["text"]) is None
            and not is_label_fragment(item)
            and not is_expiry_column(item)
            and not _contains_date(item["text"])
        ]
        if not candidates:
            return None

        # Quê quán được in trên một dòng. Sắp xếp theo x thay vì center_y
        # để box cao không đẩy cụm đầu dòng xuống sau tên tỉnh.
        value = _join_spatial_row(candidates)
    else:
        # Giữ các box giá trị nằm ngay bên phải nhãn (ví dụ ': 18',
        # 'Nguyên', 'Hồng') dù một box cao chạm cả hai dòng.
        first_row = [
            item
            for item in boxes
            if item["_left"] >= anchor_left - 8.0
            and _same_spatial_row(anchor, item)
            and not is_other_label(item)
            and not is_expiry_column(item)
        ]
        first_row_ids = {id(item) for item in first_row}
        first_line = _join_spatial_row(first_row)
        cleaned_first_line = _clean_address_text(first_line)
        if (
            cleaned_first_line
            and _spatial_label_kind(cleaned_first_line)
            == "placeOfResidence"
        ):
            numeric_matches = list(re.finditer(
                r"(?<!\d)\d{1,4}(?:/\d{1,4})?(?!\d)",
                cleaned_first_line,
            ))
            slash_matches = [
                match
                for match in numeric_matches
                if "/" in match.group(0)
            ]
            numeric_suffix = (
                slash_matches[-1]
                if slash_matches
                else numeric_matches[-1]
                if numeric_matches
                else None
            )
            cleaned_first_line = (
                cleaned_first_line[numeric_suffix.start():]
                if numeric_suffix
                else ""
            )
        first_line = cleaned_first_line or ""

        lower_candidates = [
            item
            for item in boxes
            if id(item) not in first_row_ids
            and item["_center_x"] >= 265.0
            and item["_center_y"] >= anchor_center + max(
                18.0,
                anchor_height * 0.33,
            )
            and item["_center_y"] <= anchor_center + 105.0
            and _spatial_label_kind(item["text"]) is None
            and not is_label_fragment(item)
            and not is_expiry_column(item)
            and not _contains_date(item["text"])
        ]
        second_line = ""
        if lower_candidates:
            second_anchor = min(
                lower_candidates,
                key=lambda item: item["_center_y"],
            )
            second_row = [
                item
                for item in lower_candidates
                if _same_spatial_row(second_anchor, item)
            ]
            second_line = _join_spatial_row(second_row)

        value = ", ".join(
            item
            for item in (first_line, second_line)
            if item
        )

    # Dấu chấm giữa hai đơn vị hành chính thường là dấu phẩy bị OCR sai.
    value = re.sub(
        r"(?<=[A-Za-zÀ-ỹ])\.\s*(?=[A-ZÀ-Ỹ])",
        ", ",
        value,
    )
    cleaned = _clean_address_text(value)
    if not cleaned:
        return None
    normalized = _normalize_known_administrative_names(cleaned)
    normalized = _insert_known_component_boundaries(normalized)
    normalized = _clean_address_text(normalized)
    if normalized and is_valid_address(normalized, field_name):
        return normalized
    return None


def _address_candidate_score(
    value: str | None,
    field_name: str,
    source: str,
) -> float:
    if not value or not is_valid_address(value, field_name):
        return -100.0

    text = _plain(value)
    tokens = re.findall(r"[a-z0-9]+", text)
    score = min(len(tokens), 16) * 0.20
    score += min(value.count(",") + value.count(";"), 5) * 0.25
    score += min(_diacritic_score(value), 12) * 0.04
    score += {
        "SPATIAL_OCR": 4.5,
        "RAW_TEXT_RECOVERY": 4.0,
        "FIELD_OCR": 0.5,
        "FULL_CARD_OCR": 0.3,
    }.get(source, 0.0)

    administrative_tokens = (
        "thon", "to", "kp", "khu", "khoi", "xa", "phuong",
        "huyen", "quan", "thi tran", "thi xa", "thanh pho",
    )
    score += sum(token in text for token in administrative_tokens) * 0.25

    single_tokens = sum(len(token) == 1 for token in tokens)
    score -= single_tokens * 0.9
    score -= len(re.findall(r"\d{5,}", text)) * 5.0
    score -= len(re.findall(r"\d{5,}/\d{4}", text)) * 6.0
    score -= len(
        re.findall(r"[bcdfghjklmnpqrstvwxyz]{5,}", text)
    ) * 1.5
    score -= len(
        re.findall(r"(?:[a-z]\d|\d[a-z])", text)
    ) * 1.5

    label_noise = (
        "place of", "place af", "origin", "resid", "nationality",
        "date of", "expiry", "co gia", "gia tri den",
    )
    score -= sum(noise in text for noise in label_noise) * 3.0

    # A province/city at the end is a strong address boundary. If a candidate
    # contains one earlier (or repeats it), OCR likely joined two address rows
    # in reading order. Prefer the spatially reconstructed candidate instead.
    components = [
        _plain(item)
        for item in re.split(r"[,;]", value)
        if item.strip()
    ]
    province_keys = {_plain(item) for item in CANONICAL_PROVINCES}
    province_positions = [
        index
        for index, component in enumerate(components)
        if component in province_keys
    ]
    if any(index < len(components) - 1 for index in province_positions):
        score -= 6.0
    if len(province_positions) >= 2:
        score -= 3.0
    return round(score, 4)


def select_address(
    field_name: str,
    field_value: str | None,
    full_card_value: str | None,
    raw_text: list[str] | None,
    extra_alternatives: list[str | None] | None = None,
    extra_candidates: list[str] | None = None,
    text_boxes: list[dict[str, Any]] | None = None,
    image_size: tuple[int, int] | list[int] | None = None,
) -> tuple[str | None, str]:
    def prepare(value: str | None) -> str | None:
        cleaned = _clean_address_text(value)
        if not cleaned:
            return None
        normalized = _normalize_known_administrative_names(cleaned)
        return _clean_address_text(normalized)

    raw_value = prepare(recover_address(raw_text, field_name))
    spatial_value = prepare(
        recover_spatial_address(
            text_boxes,
            field_name,
            image_size=image_size,
        )
    )
    full_value = prepare(full_card_value)
    field_clean = prepare(field_value)
    prepared_extra = [
        prepare(value)
        for value in extra_candidates or []
    ]

    candidate_pool = [
        spatial_value,
        raw_value,
        full_value,
        field_clean,
        *prepared_extra,
    ]
    candidates = [
        (spatial_value, "SPATIAL_OCR"),
        (raw_value, "RAW_TEXT_RECOVERY"),
        (full_value, "FULL_CARD_OCR"),
        (field_clean, "FIELD_OCR"),
        *[
            (candidate, "FIELD_OCR")
            for candidate in prepared_extra
            if candidate
        ],
    ]
    candidates = [
        (
            _restore_address_accents(candidate, candidate_pool)
            if candidate
            else None,
            source,
        )
        for candidate, source in candidates
    ]

    value, source = max(
        candidates,
        key=lambda item: _address_candidate_score(
            item[0],
            field_name,
            item[1],
        ),
    )
    if value and is_valid_address(value, field_name):
        raw_had_numeric_noise = any(
            re.search(r"\d{5,}/\d{4}", str(line))
            for line in raw_text or []
        )
        if (
            source == "RAW_TEXT_RECOVERY"
            and raw_had_numeric_noise
            and field_clean
            and _plain(field_clean) == _plain(value)
        ):
            value = field_clean
            source = "FIELD_OCR"
        restored = _restore_address_accents(
            value,
            [
                *candidate_pool,
            ],
        )
        restored = _restore_address_numbers(
            restored,
            extra_alternatives or [],
        )
        return (
            restored,
            source,
        )

    return None, "NOT_FOUND"


def reconcile_origin_with_residence(
    origin: str | None,
    residence: str | None,
    origin_evidence: list[str | None],
) -> tuple[str | None, str | None]:
    """Khôi phục quê quán là hậu tố của thường trú khi có bằng chứng.

    Nhiều thẻ có nơi thường trú dạng Thôn/KP cộng quê quán. Chỉ dùng
    hậu tố khi quê quán hiện tại không hợp lệ, tỉnh cuối và ít nhất một
    từ địa danh của hậu tố vẫn xuất hiện trong OCR vùng quê quán.
    """
    if is_valid_address(origin, "placeOfOrigin"):
        return origin, None
    if not residence:
        return origin, None

    components = [
        item.strip()
        for item in re.split(r"[,;]", residence)
        if item.strip()
    ]
    if len(components) < 4:
        return origin, None

    suffix = _clean_address_text(", ".join(components[1:]))
    if not suffix:
        return origin, None
    suffix = _normalize_known_administrative_names(suffix)
    suffix = _clean_address_text(suffix)
    if not is_valid_address(suffix, "placeOfOrigin"):
        return origin, None

    evidence_text = _plain(
        " ".join(str(value) for value in origin_evidence if value)
    )
    suffix_components = [
        _plain(item)
        for item in re.split(r"[,;]", suffix)
        if item.strip()
    ]
    if not suffix_components:
        return origin, None

    province = suffix_components[-1]
    if province not in evidence_text:
        return origin, None

    supporting_words = {
        word
        for component in suffix_components[:-1]
        for word in component.split()
        if len(word) >= 4
    }
    if not any(word in evidence_text for word in supporting_words):
        return origin, None

    return suffix, "RESIDENCE_SUFFIX_RECOVERY"


def looks_like_cross_field_address_leakage(
    origin: str | None,
    residence: str | None,
) -> bool:
    """Phát hiện crop cư trú thực chất đang đọc lại dòng quê quán."""
    if not origin or not residence:
        return False

    def province_of(value: str) -> str | None:
        plain_value = _plain(value)
        matches = [
            province
            for province in CANONICAL_PROVINCES
            if re.search(
                rf"\b{re.escape(_plain(province))}\s*$",
                plain_value,
            )
        ]
        return max(matches, key=len) if matches else None

    origin_province = province_of(origin)
    residence_province = province_of(residence)
    if (
        not origin_province
        or not residence_province
        or _plain(origin_province) == _plain(residence_province)
    ):
        return False

    ignored = {
        "thi", "tran", "xa", "phuong", "thanh", "pho",
        "huyen", "quan", "thon", "tdp", "kp",
    }
    origin_tokens = {
        token
        for token in re.findall(r"[a-z]+", _plain(origin))
        if token not in ignored and len(token) >= 3
    }
    residence_tokens = {
        token
        for token in re.findall(r"[a-z]+", _plain(residence))
        if token not in ignored and len(token) >= 3
    }
    if not origin_tokens or not residence_tokens:
        return False

    overlap = len(origin_tokens & residence_tokens) / min(
        len(origin_tokens),
        len(residence_tokens),
    )
    return overlap >= 0.50


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


def _looks_like_name_label_fragment(
    value: str | None,
    alternatives: list[str | None],
) -> bool:
    """Loại mẩu `Họ và` khi nhãn mờ bị parser coi là họ tên."""
    if not value:
        return False
    words = _name_key(value).split()
    has_complete_alternative = any(
        candidate and len(_name_key(candidate).split()) >= 3
        for candidate in alternatives
        if candidate != value
    )
    return bool(
        has_complete_alternative
        and len(words) == 2
        and words[0] == "HO"
        and re.fullmatch(r"VA{1,2}", words[1])
    )


def _restore_name_diacritics(
    value: str,
    gender_hint: str | None,
) -> str:
    """
    Phục hồi có kiểm soát cho họ phổ biến và tên đệm theo giới tính.

    Không dùng một từ điển chung để đoán mọi âm tiết vì cùng một chuỗi
    không dấu có thể tương ứng nhiều tên hợp lệ khác nhau.
    """
    words = value.split()
    if not words:
        return value

    surname_key = remove_accents(words[0]).upper()
    surname = (
        OCR_SURNAME_ALIASES.get(surname_key)
        or COMMON_VIETNAMESE_SURNAMES.get(surname_key)
    )
    if surname:
        words[0] = surname

    contextual_map: dict[str, str] = {}
    if gender_hint == "Nữ":
        contextual_map = FEMALE_NAME_DIACRITICS
    elif gender_hint == "Nam":
        contextual_map = MALE_NAME_DIACRITICS

    for index in range(1, len(words)):
        word_key = remove_accents(words[index]).upper()
        replacement = (
            UNAMBIGUOUS_NAME_DIACRITICS.get(word_key)
            or contextual_map.get(word_key)
        )
        if replacement:
            words[index] = replacement

    return " ".join(words)


def _name_candidate_score(
    value: str,
    source: str,
    support: int,
) -> float:
    """Chấm họ tên theo đồng thuận và cấu trúc, không theo thứ tự nguồn."""
    words = value.split()
    if not 2 <= len(words) <= 7:
        return -100.0

    surname_key = remove_accents(words[0]).upper()
    has_known_surname = bool(
        surname_key in COMMON_VIETNAMESE_SURNAMES
        or surname_key in OCR_SURNAME_ALIASES
    )

    score = {
        "FULL_CARD_OCR": 8.0,
        # Tên được phục hồi sau nhãn trên OCR toàn thẻ giữ được ngữ cảnh
        # bố cục tốt hơn nhiều biến thể crop cùng lặp một lỗi ký tự.
        "RAW_TEXT_RECOVERY": 6.0,
        "FIELD_OCR": 2.25,
    }.get(source, 1.0)
    score += min(support, 4) * 1.25
    score += 6.0 if has_known_surname else -3.0
    if 3 <= len(words) <= 4:
        score += 4.0
    elif len(words) == 5:
        score += 1.0
    elif len(words) >= 6:
        score -= 2.0
    else:
        score += 0.5
    score += min(_diacritic_score(value), 10) * 0.12

    # Các mẩu nhãn mờ như ``Ho va Jen / FW ratie`` có thể đi ngay sau
    # số CCCD và trước đây bị xem là tên. Chỉ phạt khi có từ hai token
    # cùng giống nhãn; một tên riêng đơn lẻ không bị loại vì fuzzy match.
    label_words = ("TEN", "FULL", "NAME", "NGAY", "SINH", "DATE", "BIRTH")
    fuzzy_label_tokens = sum(
        any(
            SequenceMatcher(
                None,
                remove_accents(word).upper(),
                label,
                ).ratio() >= 0.82
            for label in label_words
        )
        for word in words[1:]
    )
    if fuzzy_label_tokens >= 2:
        score -= fuzzy_label_tokens * 4.0

    unexplained_short_words = sum(
        len(remove_accents(word)) <= 2
        and remove_accents(word).upper() not in {
            "LE", "LY", "DO", "HO", "VU", "VO", "TA",
        }
        for word in words
    )
    score -= unexplained_short_words * 2.0
    return round(score, 4)


def select_full_name(
    field_value: str | None,
    full_card_value: str | None,
    raw_text: list[str] | None,
    gender_hint: str | None = None,
    extra_evidence: list[str] | None = None,
) -> tuple[str | None, str]:
    """
    Chọn họ tên bằng đồng thuận giữa OCR toàn thẻ, OCR vùng và mọi biến
    thể ảnh của vùng tên. Cách chấm này ngăn mẩu nhãn mờ hai từ thắng một
    họ tên ba/bốn từ được nhiều biến thể đọc giống nhau.
    """

    full_name = normalize_full_name(full_card_value)
    raw_name = recover_full_name(raw_text)
    field_name = normalize_full_name(field_value)
    candidates: list[tuple[str | None, str]] = [
        (full_name, "FULL_CARD_OCR"),
        (raw_name, "RAW_TEXT_RECOVERY"),
        (field_name, "FIELD_OCR"),
    ]

    for value in extra_evidence or []:
        candidates.append((normalize_full_name(value), "FIELD_OCR"))

    alternatives = [candidate for candidate, _ in candidates]
    candidates = [
        (candidate, source)
        for candidate, source in candidates
        if candidate
        and not _looks_like_name_label_fragment(candidate, alternatives)
    ]
    if not candidates:
        return None, "NOT_FOUND"

    support: dict[str, int] = {}
    for candidate, _ in candidates:
        key = _name_key(candidate)
        support[key] = support.get(key, 0) + 1

    selected_value, selected_source = max(
        candidates,
        key=lambda item: (
            _name_candidate_score(
                item[0],
                item[1],
                support.get(_name_key(item[0]), 1),
            ),
            _diacritic_score(item[0]),
        ),
    )

    # Nếu raw chỉ lệch đúng một ký tự còn ít nhất hai crop độc lập cùng
    # đọc một phương án rất gần, dùng đồng thuận crop để sửa ký tự đó.
    # Ngưỡng cao và yêu cầu cùng số từ ngăn tên bị cắt cụt hoặc một lỗi
    # khác hẳn (ví dụ ``NAM`` -> ``VGAY``) lấn át dòng tên có nhãn.
    if (
        selected_source == "RAW_TEXT_RECOVERY"
        and support.get(_name_key(selected_value), 1) == 1
    ):
        selected_words = _name_key(selected_value).split()
        close_consensus = [
            (candidate, source)
            for candidate, source in candidates
            if support.get(_name_key(candidate), 1) >= 2
            and len(_name_key(candidate).split()) == len(selected_words)
            and SequenceMatcher(
                None,
                _name_key(candidate),
                _name_key(selected_value),
            ).ratio() >= 0.92
        ]
        if close_consensus:
            selected_value, selected_source = max(
                close_consensus,
                key=lambda item: (
                    support.get(_name_key(item[0]), 1),
                    _diacritic_score(item[0]),
                ),
            )

    # Trong nhóm cùng chữ gốc, lấy bản nhiều dấu nhất nhưng giữ ưu tiên
    # provenance FULL -> RAW -> FIELD khi lượng dấu bằng nhau.
    source_priority = {
        "FULL_CARD_OCR": 3,
        "RAW_TEXT_RECOVERY": 2,
        "FIELD_OCR": 1,
    }
    equivalent_candidates = [
        (candidate, source)
        for candidate, source in candidates
        if _name_key(candidate) == _name_key(selected_value)
    ]
    selected_value, selected_source = max(
        equivalent_candidates,
        key=lambda item: (
            _diacritic_score(item[0]),
            source_priority.get(item[1], 0),
        ),
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

    # Nhãn quốc tịch có thể bị hỏng nhưng giá trị ``Việt Nam`` vẫn còn.
    # Chỉ dùng lần quét fuzzy này sau các nguồn có cấu trúc để giữ đúng
    # provenance khi full-card/field OCR đã đọc được giá trị.
    for line in raw_text or []:
        raw_value = normalize_nationality(str(line))
        if raw_value:
            return raw_value, "RAW_TEXT_RECOVERY"

    return None, "NOT_FOUND"


def _build_date(day: int, month: int, year: int) -> str | None:
    try:
        parsed = datetime(year=year, month=month, day=day)
    except ValueError:
        return None
    if not 1900 <= parsed.year <= 2100:
        return None
    return parsed.strftime("%d/%m/%Y")


def _date_candidates_from_value(value: str | None) -> list[str]:
    """Lấy mọi ngày hợp lệ trong một chuỗi OCR thay vì ứng viên đầu tiên.

    Hỗ trợ ngày dính với số rác (``4.96724/03/2035``) và ngày có thừa
    một ký tự số (``05/0172035``). Việc chọn ứng viên cuối cùng vẫn do
    ``select_date`` chấm theo ngày sinh và mốc tuổi CCCD.
    """
    if not value:
        return []

    normalized = normalize_ocr_digits(str(value))
    candidates: list[str] = []

    def add(candidate: str | None) -> None:
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    add(normalize_date(normalized))

    # Không dùng lookaround chữ số ở đây: ảnh thật có thể dính một cụm
    # số rác ngay trước ngày nhưng bản thân DD/MM/YYYY vẫn hoàn chỉnh.
    for match in re.finditer(
        r"(\d{1,2})\s*[/.-]\s*(\d{1,2})\s*[/.-]\s*(\d{4})",
        normalized,
    ):
        add(_build_date(*(int(item) for item in match.groups())))

    numeric_groups = re.findall(
        r"\d(?:[\d\s/.-]{4,12}\d)?",
        normalized,
    )
    for group in numeric_groups:
        digits = re.sub(r"\D", "", group)
        digit_variants: list[str] = []
        if len(digits) == 8:
            digit_variants.append(digits)
        elif len(digits) == 9:
            digit_variants.extend(
                digits[:index] + digits[index + 1:]
                for index in range(9)
            )

        for item in digit_variants:
            add(_build_date(
                int(item[:2]),
                int(item[2:4]),
                int(item[4:]),
            ))

    return candidates


def _expiry_milestone_from_evidence(
    date_of_birth: str | None,
    evidence: list[str],
) -> tuple[str | None, float]:
    """Khôi phục hạn dùng chỉ khi crop hạn dùng còn bằng chứng số đủ mạnh."""
    if not date_of_birth:
        return None, 0.0
    try:
        birth = datetime.strptime(date_of_birth, "%d/%m/%Y")
    except ValueError:
        return None, 0.0

    best_value: str | None = None
    best_score = 0.0
    for age in (25, 40, 60):
        expiry_year = birth.year + age
        # Mặt thẻ trong pipeline là CCCD gắn chip; mốc đã trôi qua trước
        # thời điểm phát hành loại thẻ này không thể là hạn in trên thẻ.
        if expiry_year < 2021:
            continue
        expected = birth.replace(year=expiry_year).strftime("%d/%m/%Y")
        expected_digits = re.sub(r"\D", "", expected)
        expected_year = expected_digits[-4:]

        for value in evidence:
            normalized = normalize_ocr_digits(str(value))
            groups = re.findall(r"\d(?:[\d\s/.-]{4,14}\d)?", normalized)
            for group in groups:
                digits = re.sub(r"\D", "", group)
                if len(digits) < 6:
                    continue

                # Chuỗi dài do nhiều box dính nhau được xét theo từng cửa
                # sổ gần độ dài ngày, thay vì so cả cụm số hỗn hợp.
                windows = [digits]
                if len(digits) > 9:
                    windows.extend(
                        digits[start:start + size]
                        for size in (7, 8, 9)
                        for start in range(0, len(digits) - size + 1)
                    )

                for window in windows:
                    similarity = SequenceMatcher(
                        None,
                        window,
                        expected_digits,
                    ).ratio()
                    has_year = expected_year in window
                    if not has_year and similarity < 0.78:
                        continue
                    score = similarity + (0.25 if has_year else 0.0)
                    if score > best_score:
                        best_value = expected
                        best_score = score

    return best_value, best_score


def select_date(
    field_name: str,
    field_value: str | None,
    full_card_value: str | None,
    raw_text: list[str] | None,
    date_of_birth: str | None = None,
    extra_evidence: list[str] | None = None,
) -> tuple[str | None, str]:
    if field_name == "dateOfExpiry":
        raw_labeled = recover_labeled_date(raw_text, field_name)
        candidates: list[tuple[str | None, str, float]] = [
            (raw_labeled, "RAW_TEXT_RECOVERY", 3.0),
        ]
        candidates.extend(
            (candidate, "FULL_CARD_OCR", 2.5)
            for candidate in _date_candidates_from_value(full_card_value)
        )
        candidates.extend(
            (candidate, "FIELD_OCR", 2.25)
            for candidate in _date_candidates_from_value(field_value)
        )
        candidates.extend(
            (candidate, "FIELD_OCR", 2.5)
            for value in extra_evidence or []
            for candidate in _date_candidates_from_value(value)
        )
        candidates.extend(
            (candidate, "RAW_TEXT_RECOVERY", 1.5)
            for line in raw_text or []
            for candidate in _date_candidates_from_value(str(line))
            if candidate != date_of_birth
        )

        parsed_birth: datetime | None = None
        if date_of_birth:
            try:
                parsed_birth = datetime.strptime(
                    date_of_birth,
                    "%d/%m/%Y",
                )
            except ValueError:
                parsed_birth = None

        scored: list[tuple[float, str, str]] = []
        for candidate, source, base_score in candidates:
            if not candidate:
                continue
            try:
                parsed = datetime.strptime(candidate, "%d/%m/%Y")
            except ValueError:
                continue

            score = base_score
            if parsed_birth is not None:
                age = parsed.year - parsed_birth.year
                if age <= 0 or age > 100:
                    score -= 20.0
                if age in {25, 40, 60}:
                    score += 4.0
                    # Hạn CCCD theo mốc tuổi rơi vào ngày sinh. Chỉ sửa
                    # ngày bị mất một chữ số khi tháng và mốc tuổi khớp.
                    if parsed.month == parsed_birth.month:
                        if parsed.day != parsed_birth.day:
                            parsed = parsed.replace(day=parsed_birth.day)
                            candidate = parsed.strftime("%d/%m/%Y")
                        score += 6.0
                    else:
                        # Hạn CCCD tại mốc 25/40/60 tuổi phải rơi vào
                        # ngày sinh; tháng khác là ứng viên OCR đã xóa
                        # nhầm một chữ số, không được ưu tiên chỉ vì năm.
                        score -= 8.0
                elif (
                    parsed.day == parsed_birth.day
                    and parsed.month == parsed_birth.month
                ):
                    score += 3.0
            scored.append((score, candidate, source))

        milestone_evidence = [
            str(value)
            for value in (
                [full_card_value, field_value]
                + list(extra_evidence or [])
            )
            if value
        ]
        milestone_evidence.extend(
            str(line)
            for line in raw_text or []
            if line and _looks_like_expiry_label(str(line))
        )
        milestone, evidence_score = _expiry_milestone_from_evidence(
            date_of_birth,
            milestone_evidence,
        )
        existing_values = {
            candidate
            for _, candidate, _ in scored
        }
        if (
            milestone
            and evidence_score >= 0.78
            and milestone not in existing_values
        ):
            scored.append((
                13.0 + evidence_score,
                milestone,
                "ID_MILESTONE_RECOVERY(FIELD_OCR)",
            ))

        if scored:
            _, value, source = max(scored, key=lambda item: item[0])
            return value, source
        return None, "NOT_FOUND"

    candidates = [
        (recover_labeled_date(raw_text, field_name), "RAW_TEXT_RECOVERY", 4.0),
        *[
            (candidate, "FULL_CARD_OCR", 3.0)
            for candidate in _date_candidates_from_value(full_card_value)
        ],
        *[
            (candidate, "FIELD_OCR", 3.25)
            for candidate in _date_candidates_from_value(field_value)
        ],
        *[
            (candidate, "FIELD_OCR", 3.5)
            for value in extra_evidence or []
            for candidate in _date_candidates_from_value(value)
        ],
    ]
    available = [candidate for candidate in candidates if candidate[0]]
    if available:
        value, source, _ = max(available, key=lambda item: item[2])
        return value, source

    return None, "NOT_FOUND"


def fuse_ocr_data(
    full_card_data: dict[str, Any] | None,
    field_data: dict[str, Any] | None,
    raw_text: list[str] | None = None,
    field_results: dict[str, Any] | None = None,
    text_boxes: list[dict[str, Any]] | None = None,
    image_size: tuple[int, int] | list[int] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    full_card = full_card_data or {}
    fields = field_data or {}
    evidence = {
        field_name: collect_field_evidence(field_results, field_name)
        for field_name in FIELD_NAMES
    }

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

    gender_hint = (
        recover_gender(raw_text)
        or normalize_gender(full_card.get("gender"))
        or normalize_gender(fields.get("gender"))
    )

    date_of_birth, date_of_birth_source = select_date(
        "dateOfBirth",
        fields.get("dateOfBirth"),
        full_card.get("dateOfBirth"),
        raw_text,
        extra_evidence=evidence["dateOfBirth"],
    )

    id_number, id_source = select_id_number(
        full_card.get("idNumber"),
        fields.get("idNumber"),
        raw_text,
        date_of_birth=date_of_birth,
        gender=gender_hint,
    )
    result["idNumber"] = id_number
    sources["idNumber"] = id_source

    date_of_birth, date_of_birth_source = reconcile_birth_date_with_id(
        identifier=id_number,
        selected_date=date_of_birth,
        selected_source=date_of_birth_source,
        field_value=fields.get("dateOfBirth"),
        full_card_value=full_card.get("dateOfBirth"),
        raw_text=raw_text,
        extra_evidence=evidence["dateOfBirth"],
    )

    _, id_gender_hint = decode_id_demographics(id_number)
    effective_gender_hint = gender_hint or id_gender_hint

    full_name, source = select_full_name(
        fields.get("fullName"),
        full_card.get("fullName"),
        raw_text,
        gender_hint=effective_gender_hint,
        extra_evidence=evidence["fullName"],
    )
    result["fullName"] = full_name
    sources["fullName"] = source

    result["dateOfBirth"] = date_of_birth
    sources["dateOfBirth"] = date_of_birth_source

    gender, source = select_gender(
        fields.get("gender"),
        full_card.get("gender"),
        raw_text,
    )
    if not gender and id_gender_hint:
        gender = id_gender_hint
        source = "ID_STRUCTURE_RECOVERY"
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
        other_field_name = (
            "placeOfResidence"
            if field_name == "placeOfOrigin"
            else "placeOfOrigin"
        )
        value, source = select_address(
            field_name,
            fields.get(field_name),
            full_card.get(field_name),
            raw_text,
            extra_alternatives=[
                fields.get(other_field_name),
            ],
            extra_candidates=evidence[field_name],
            text_boxes=text_boxes,
            image_size=image_size,
        )
        result[field_name] = value
        sources[field_name] = source

    reconciled_origin, reconciliation_source = (
        reconcile_origin_with_residence(
            result.get("placeOfOrigin"),
            result.get("placeOfResidence"),
            [
                full_card.get("placeOfOrigin"),
                fields.get("placeOfOrigin"),
                *evidence["placeOfOrigin"],
                *(raw_text or []),
            ],
        )
    )
    if reconciliation_source:
        result["placeOfOrigin"] = reconciled_origin
        sources["placeOfOrigin"] = reconciliation_source

    if (
        sources.get("placeOfResidence") == "FIELD_OCR"
        and looks_like_cross_field_address_leakage(
            result.get("placeOfOrigin"),
            result.get("placeOfResidence"),
        )
    ):
        result["placeOfResidence"] = None
        sources["placeOfResidence"] = "NOT_FOUND"

    expiry_cross_field_evidence = [
        value
        for field_name in ("placeOfOrigin", "placeOfResidence")
        for value in evidence[field_name]
        if re.search(r"\d", str(value))
    ]
    expiry, source = select_date(
        "dateOfExpiry",
        fields.get("dateOfExpiry"),
        full_card.get("dateOfExpiry"),
        raw_text,
        date_of_birth=date_of_birth,
        extra_evidence=[
            *evidence["dateOfExpiry"],
            *expiry_cross_field_evidence,
        ],
    )
    result["dateOfExpiry"] = expiry
    sources["dateOfExpiry"] = source

    return result, sources
