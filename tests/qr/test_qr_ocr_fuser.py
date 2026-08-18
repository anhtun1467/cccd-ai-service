from __future__ import annotations

from app.modules.ocr.validator import CCCDValidator
from app.modules.qr.qr_ocr_fuser import (
    build_qr_reference_data,
    fuse_qr_data,
    select_qr_field_ocr_skips,
)


def _qr_data() -> dict[str, str]:
    return {
        "idNumber": "042096015766",
        "fullName": "ĐINH XUÂN HOÀNG",
        "dateOfBirth": "24/11/1996",
        "gender": "Nam",
        "placeOfResidence": "Lâm Trung Thủy, Đức Thọ, Hà Tĩnh",
    }


def test_qr_preserves_diacritics_and_supersedes_valid_conflicting_ocr() -> None:
    ocr_data = {
        "idNumber": "042096015767",
        "fullName": "DINH XUAN HOANG",
        "dateOfBirth": "24/11/1996",
        "gender": "Nam",
        "placeOfResidence": None,
    }
    sources = {field_name: "FULL_CARD_OCR" for field_name in ocr_data}

    merged, merged_sources, diagnostics = fuse_qr_data(
        ocr_data=ocr_data,
        ocr_sources=sources,
        qr_data=_qr_data(),
        validator=CCCDValidator(),
    )

    assert merged["idNumber"] == "042096015766"
    assert merged["fullName"] == "ĐINH XUÂN HOÀNG"
    assert merged_sources["idNumber"] == "CCCD_QR"
    assert diagnostics["agreementFields"] == [
        "fullName",
        "dateOfBirth",
        "gender",
    ]
    assert diagnostics["conflicts"] == [
        {
            "field": "idNumber",
            "ocrSource": "FULL_CARD_OCR",
            "resolution": "CCCD_QR",
            "requiresReview": True,
        }
    ]


def test_field_ocr_is_skipped_only_when_full_card_does_not_conflict() -> None:
    full_card_data = {
        **_qr_data(),
        "idNumber": "042096015767",
        "fullName": "DINH XUAN HOANG",
    }
    validator = CCCDValidator()

    skipped = select_qr_field_ocr_skips(
        qr_data=_qr_data(),
        full_card_data=full_card_data,
        validator=validator,
    )
    reference = build_qr_reference_data(
        full_card_data=full_card_data,
        qr_data=_qr_data(),
        validator=validator,
    )

    assert "idNumber" not in skipped
    assert "fullName" in skipped
    assert "dateOfBirth" in skipped
    assert "gender" in skipped
    assert "placeOfResidence" in skipped
    assert reference["idNumber"] == "042096015766"


def test_qr_residence_difference_is_advisory_and_skips_redundant_ocr() -> None:
    qr_data = _qr_data()
    ocr_data = {
        **qr_data,
        "placeOfResidence": (
            "Phường Bến Nghé, Quận 1, Thành phố Hồ Chí Minh"
        ),
    }
    validator = CCCDValidator()

    merged, _, diagnostics = fuse_qr_data(
        ocr_data=ocr_data,
        ocr_sources={field_name: "FULL_CARD_OCR" for field_name in ocr_data},
        qr_data=qr_data,
        validator=validator,
    )
    skipped = select_qr_field_ocr_skips(
        qr_data=qr_data,
        full_card_data=ocr_data,
        validator=validator,
    )

    assert merged["placeOfResidence"] == qr_data["placeOfResidence"]
    assert diagnostics["conflicts"] == [
        {
            "field": "placeOfResidence",
            "ocrSource": "FULL_CARD_OCR",
            "resolution": "CCCD_QR",
            "requiresReview": False,
        }
    ]
    assert "placeOfResidence" in skipped
