from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from app.modules.ocr.result_fuser import (
    fuse_ocr_data,
    normalize_date,
    normalize_nationality,
)


def test_normalize_missing_date_separator() -> None:
    assert normalize_date(
        "24/0311995"
    ) == "24/03/1995"


def test_normalize_date_digits_only() -> None:
    assert normalize_date(
        "24031995"
    ) == "24/03/1995"


def test_normalize_nationality() -> None:
    assert normalize_nationality(
        "Viet Nam"
    ) == "Viet Nam"

    assert normalize_nationality(
        "Mre"
    ) is None


def test_fuse_bad_field_data_with_full_card() -> None:
    full_card_data = {
        "idNumber": "001095014159",
        "fullName": "NGUYEN HOANG NAM",
        "dateOfBirth": None,
        "gender": "Nam",
        "nationality": "Viet Nam",
    }

    field_data = {
        "idNumber": None,
        "fullName": "S NO VA TEN HO",
        "dateOfBirth": None,
        "gender": None,
        "nationality": "Mre",
    }

    raw_text = [
        "Ho va ten / Full name:",
        "NGUYEN HOANG NAM",
        "Ngay sinh / Date of birth: 24/0311995",
        "Quoc tich / Nationality: Viet Nam",
    ]

    result, sources = fuse_ocr_data(
        full_card_data=full_card_data,
        field_data=field_data,
        raw_text=raw_text,
    )

    assert result["fullName"] == "NGUYEN HOANG NAM"
    assert result["dateOfBirth"] == "24/03/1995"
    assert result["nationality"] == "Viet Nam"

    assert sources["fullName"] == "FULL_CARD_OCR"
    assert sources["dateOfBirth"] == "RAW_TEXT_RECOVERY"
    assert sources["nationality"] == "FULL_CARD_OCR"


if __name__ == "__main__":
    test_normalize_missing_date_separator()
    test_normalize_date_digits_only()
    test_normalize_nationality()
    test_fuse_bad_field_data_with_full_card()

    print("Tất cả test result_fuser đã PASS.")
