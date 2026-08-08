from app.modules.ocr.field_cropper import CCCDFieldCropper
from app.modules.ocr.field_ocr_service import FieldOCRService


def test_value_regions_stay_inside_canonical_card() -> None:
    cropper = CCCDFieldCropper()

    assert set(cropper.VALUE_REGIONS) == set(cropper.FIELD_REGIONS) - {
        "portrait"
    }
    for region in cropper.VALUE_REGIONS.values():
        cropper.validate_region(region)
    for region in cropper.TIGHT_VALUE_REGIONS.values():
        cropper.validate_region(region)

    expiry = cropper.TIGHT_VALUE_REGIONS["dateOfExpiry"]
    wide_expiry = cropper.VALUE_REGIONS["dateOfExpiry"]
    assert wide_expiry.x1 <= expiry.x1 < expiry.x2 <= wide_expiry.x2
    assert wide_expiry.y1 <= expiry.y1 < expiry.y2 <= wide_expiry.y2


def test_name_candidates_merge_accents_word_by_word() -> None:
    selected = FieldOCRService.select_best_field_candidate(
        field_name="fullName",
        candidates=[
            {
                "variant": "value_processed",
                "value": "DƯƠNG THI HUONG HIỆP",
                "confidence": 0.55,
            },
            {
                "variant": "value_raw",
                "value": "DUONG THỊ HƯƠNG HIEP",
                "confidence": 0.51,
            },
            {
                "variant": "wide_raw",
                "value": "DUONG THI HUONG HIEP",
                "confidence": 0.70,
            },
        ],
    )

    assert selected["value"] == "DƯƠNG THỊ HƯƠNG HIỆP"
    assert "accent_consensus" in selected["variant"]


def test_invalid_ocr_year_requests_another_image_variant() -> None:
    assert not FieldOCRService.can_stop_field_retries(
        field_name="dateOfBirth",
        candidates=[{"value": "17/10/1801"}],
    )
    assert FieldOCRService.can_stop_field_retries(
        field_name="dateOfBirth",
        candidates=[{"value": "17/10/1991"}],
    )


def test_blurry_gender_and_nationality_confusions_are_normalized() -> None:
    assert FieldOCRService.clean_gender("NÔ") == "Nữ"
    assert FieldOCRService.clean_nationality("Vie Naii") == "Việt Nam"


def test_field_date_rejects_impossible_cccd_year() -> None:
    service = FieldOCRService()

    assert service.clean_date("17/10/1801") is None
    assert service.clean_date("17/10/1991") == "17/10/1991"
