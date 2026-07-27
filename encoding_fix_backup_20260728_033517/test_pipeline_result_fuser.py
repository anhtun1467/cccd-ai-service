from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT_DIR),
    )


from app.modules.ocr.result_fuser import fuse_ocr_data


def test_pipeline_fusion_data() -> None:
    full_card_data = {
        "idNumber": None,
        "fullName": "NGUYEN HOANG NAM",
        "dateOfBirth": None,
        "gender": None,
        "nationality": "Viet Nam",
        "placeOfOrigin": None,
        "placeOfResidence": None,
        "dateOfExpiry": None,
    }

    field_data = {
        "idNumber": None,
        "fullName": "S NO VA TEN HO",
        "dateOfBirth": None,
        "gender": None,
        "nationality": "Mre",
        "placeOfOrigin": None,
        "placeOfResidence": None,
        "dateOfExpiry": None,
    }

    raw_text = [
        "CAN CUOC CONG DAN",
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

    assert result["fullName"] == (
        "NGUYEN HOANG NAM"
    )

    assert result["dateOfBirth"] == (
        "24/03/1995"
    )

    assert result["nationality"] == (
        "Viet Nam"
    )

    assert sources["fullName"] == (
        "FULL_CARD_OCR"
    )

    assert sources["dateOfBirth"] == (
        "RAW_TEXT_RECOVERY"
    )

    assert sources["nationality"] == (
        "FULL_CARD_OCR"
    )


if __name__ == "__main__":
    test_pipeline_fusion_data()

    print("=" * 68)
    print("TEST TÍCH H?P RESULT_FUSER ÐÃ PASS")
    print("=" * 68)

