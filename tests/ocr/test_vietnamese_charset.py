from __future__ import annotations

import unicodedata

from app.modules.ocr.vietnamese_charset import (
    DIGITS,
    FIELD_ALLOWLISTS,
    TEMPLATE_CHARACTERS,
    VIETNAMESE_LETTERS_LOWER,
    VIETNAMESE_LETTERS_UPPER,
    VIETNAMESE_OCR_ALLOWLIST,
    field_allowlist,
    normalize_nfc,
    unsupported_characters,
    visual_candidates,
)


REQUESTED_SAMPLE = "aăâáàãạấầậẫắặ"


def test_library_contains_all_89_lower_and_upper_vietnamese_letters() -> None:
    assert len(VIETNAMESE_LETTERS_LOWER) == 89
    assert len(set(VIETNAMESE_LETTERS_LOWER)) == 89
    assert len(VIETNAMESE_LETTERS_UPPER) == 89
    assert len(set(VIETNAMESE_LETTERS_UPPER)) == 89

    assert set(REQUESTED_SAMPLE) <= set(VIETNAMESE_LETTERS_LOWER)
    assert set(REQUESTED_SAMPLE.upper()) <= set(VIETNAMESE_LETTERS_UPPER)
    assert set(DIGITS) <= set(TEMPLATE_CHARACTERS)


def test_every_vietnamese_letter_is_precomposed_nfc() -> None:
    for character in VIETNAMESE_LETTERS_LOWER + VIETNAMESE_LETTERS_UPPER:
        assert len(character) == 1
        assert unicodedata.normalize("NFC", character) == character

    assert normalize_nfc("a\N{COMBINING BREVE}\N{COMBINING ACUTE ACCENT}") == "ắ"
    assert normalize_nfc(
        "a\N{COMBINING CIRCUMFLEX ACCENT}\N{COMBINING DOT BELOW}"
    ) == "ậ"


def test_field_allowlists_separate_numeric_and_vietnamese_text() -> None:
    assert FIELD_ALLOWLISTS["idNumber"] == "0123456789 "
    assert set(FIELD_ALLOWLISTS["dateOfBirth"]) <= set("0123456789/.- ")
    assert set(REQUESTED_SAMPLE) <= set(FIELD_ALLOWLISTS["fullName"])
    assert field_allowlist("unknown") == VIETNAMESE_OCR_ALLOWLIST


def test_visual_candidates_stay_in_safe_character_families() -> None:
    lower_a = set(visual_candidates("a", "fullName"))
    upper_a = set(visual_candidates("A", "fullName"))

    assert set(REQUESTED_SAMPLE) <= lower_a
    assert set(REQUESTED_SAMPLE.upper()) <= upper_a
    assert not any(character.isupper() for character in lower_a)
    assert not any(character.islower() for character in upper_a)
    assert visual_candidates("Đ", "fullName") == ("D", "Đ")
    assert visual_candidates("O", "idNumber") == tuple(DIGITS)


def test_unsupported_characters_is_nfc_aware() -> None:
    assert unsupported_characters("ĐẶNG THỊ MÂY, 0123456789") == ()
    assert unsupported_characters("Việt Nam €") == ("€",)
