from __future__ import annotations

import unicodedata
from collections.abc import Iterable


# Thứ tự dấu thanh: không dấu, huyền, sắc, hỏi, ngã, nặng.
TONE_MARKS: tuple[str, ...] = (
    "",
    "\N{COMBINING GRAVE ACCENT}",
    "\N{COMBINING ACUTE ACCENT}",
    "\N{COMBINING HOOK ABOVE}",
    "\N{COMBINING TILDE}",
    "\N{COMBINING DOT BELOW}",
)

VOWEL_SHAPES: tuple[str, ...] = (
    "a",
    "ă",
    "â",
    "e",
    "ê",
    "i",
    "o",
    "ô",
    "ơ",
    "u",
    "ư",
    "y",
)

VOWEL_FAMILIES: dict[str, tuple[str, ...]] = {
    "a": ("a", "ă", "â"),
    "e": ("e", "ê"),
    "i": ("i",),
    "o": ("o", "ô", "ơ"),
    "u": ("u", "ư"),
    "y": ("y",),
}

VIETNAMESE_CONSONANTS_LOWER = "bcdđghklmnpqrstvx"
DIGITS = "0123456789"
ASCII_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Các dấu có thể xuất hiện trong dữ liệu mặt trước CCCD. Khoảng trắng được
# giữ ở allowlist OCR nhưng không tạo template ảnh riêng.
OCR_PUNCTUATION = " /.,:;-'()"
EXTENDED_PUNCTUATION = "–—_[]{}+&#@"


def _deduplicate(values: Iterable[str]) -> str:
    return "".join(dict.fromkeys(values))


def _tone_variants(vowel: str) -> tuple[str, ...]:
    return tuple(
        unicodedata.normalize("NFC", f"{vowel}{tone}")
        for tone in TONE_MARKS
    )


VIETNAMESE_VOWELS_LOWER: tuple[str, ...] = tuple(
    character
    for shape in VOWEL_SHAPES
    for character in _tone_variants(shape)
)

VIETNAMESE_LETTERS_LOWER = _deduplicate(
    (*VIETNAMESE_VOWELS_LOWER, *VIETNAMESE_CONSONANTS_LOWER)
)
VIETNAMESE_LETTERS_UPPER = VIETNAMESE_LETTERS_LOWER.upper()
VIETNAMESE_LETTERS = _deduplicate(
    VIETNAMESE_LETTERS_UPPER + VIETNAMESE_LETTERS_LOWER
)

# Bao gồm đủ 89 chữ thường, 89 chữ hoa, 10 chữ số và dấu câu dùng trên CCCD.
VIETNAMESE_OCR_ALLOWLIST = _deduplicate(
    DIGITS
    + ASCII_LETTERS
    + VIETNAMESE_LETTERS_UPPER
    + VIETNAMESE_LETTERS_LOWER
    + OCR_PUNCTUATION
)

NUMERIC_OCR_ALLOWLIST = DIGITS + "/.- "

FIELD_ALLOWLISTS: dict[str, str] = {
    "idNumber": DIGITS + " ",
    "dateOfBirth": NUMERIC_OCR_ALLOWLIST,
    "dateOfExpiry": NUMERIC_OCR_ALLOWLIST,
    "fullName": _deduplicate(VIETNAMESE_LETTERS + ASCII_LETTERS + " -'"),
    "gender": _deduplicate(VIETNAMESE_LETTERS + ASCII_LETTERS + " "),
    "nationality": _deduplicate(VIETNAMESE_LETTERS + ASCII_LETTERS + " "),
    "placeOfOrigin": VIETNAMESE_OCR_ALLOWLIST,
    "placeOfResidence": VIETNAMESE_OCR_ALLOWLIST,
}

# Không tạo template cho khoảng trắng. Các ký tự còn lại đều có thể được
# render thành ảnh và dùng để so khớp với vùng ký tự cắt từ OCR box.
TEMPLATE_CHARACTERS = _deduplicate(
    VIETNAMESE_LETTERS
    + ASCII_LETTERS
    + DIGITS
    + OCR_PUNCTUATION.replace(" ", "")
    + EXTENDED_PUNCTUATION
)


def normalize_nfc(value: str) -> str:
    """Chuẩn hóa chuỗi về Unicode NFC để mỗi chữ có dấu là một ký tự."""
    return unicodedata.normalize("NFC", str(value))


def unsupported_characters(value: str) -> tuple[str, ...]:
    """Trả về các ký tự ngoài thư viện, không tính khoảng trắng xuống dòng."""
    supported = set(
        VIETNAMESE_LETTERS
        + DIGITS
        + OCR_PUNCTUATION
        + EXTENDED_PUNCTUATION
        + "\r\n\t"
    )
    return tuple(
        dict.fromkeys(
            character
            for character in normalize_nfc(value)
            if character not in supported
        )
    )


def strip_vietnamese_marks(value: str) -> str:
    """Bỏ dấu thanh/dấu phụ nhưng vẫn xử lý riêng cặp đ/D."""
    decomposed = unicodedata.normalize("NFD", str(value))
    plain = "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    )
    return plain.replace("đ", "d").replace("Đ", "D")


def _build_visual_families() -> dict[str, tuple[str, ...]]:
    families: dict[str, tuple[str, ...]] = {}
    for shapes in VOWEL_FAMILIES.values():
        lower = tuple(
            character
            for shape in shapes
            for character in _tone_variants(shape)
        )
        upper = tuple(character.upper() for character in lower)
        for character in lower:
            families[character] = lower
        for character in upper:
            families[character] = upper

    families["d"] = ("d", "đ")
    families["đ"] = ("d", "đ")
    families["D"] = ("D", "Đ")
    families["Đ"] = ("D", "Đ")
    return families


VISUAL_FAMILIES = _build_visual_families()


def visual_candidates(
    character: str,
    field_name: str | None = None,
) -> tuple[str, ...]:
    """
    Lấy các ký tự hợp lệ để đối chiếu hình dáng.

    Trường số chỉ so giữa 0-9. Trường chữ chỉ so trong cùng họ nguyên âm
    (a/ă/â, e/ê, o/ô/ơ, u/ư...) hoặc d/đ, không tự đổi phụ âm khác.
    """
    normalized = normalize_nfc(character)
    if len(normalized) != 1:
        return ()

    if field_name in {"idNumber", "dateOfBirth", "dateOfExpiry"}:
        if normalized.isdigit() or normalized in "OoQqDdIiLlSsBbGg":
            return tuple(DIGITS)
        return (normalized,)

    return VISUAL_FAMILIES.get(normalized, (normalized,))


def field_allowlist(field_name: str | None) -> str:
    """Trả về allowlist riêng của field hoặc bảng đầy đủ tiếng Việt."""
    if not field_name:
        return VIETNAMESE_OCR_ALLOWLIST
    return FIELD_ALLOWLISTS.get(field_name, VIETNAMESE_OCR_ALLOWLIST)


__all__ = [
    "DIGITS",
    "ASCII_LETTERS",
    "EXTENDED_PUNCTUATION",
    "FIELD_ALLOWLISTS",
    "NUMERIC_OCR_ALLOWLIST",
    "OCR_PUNCTUATION",
    "TEMPLATE_CHARACTERS",
    "TONE_MARKS",
    "VIETNAMESE_LETTERS",
    "VIETNAMESE_LETTERS_LOWER",
    "VIETNAMESE_LETTERS_UPPER",
    "VIETNAMESE_OCR_ALLOWLIST",
    "VIETNAMESE_VOWELS_LOWER",
    "field_allowlist",
    "normalize_nfc",
    "strip_vietnamese_marks",
    "unsupported_characters",
    "visual_candidates",
]
