from __future__ import annotations

from app.modules.ocr.card_side_classifier import classify_cccd_side


def test_front_side_is_recognized_from_fixed_labels() -> None:
    result = classify_cccd_side({
        "normalizedText": [
            "CĂN CƯỚC CÔNG DÂN",
            "Citizen Identity Card",
            "Họ và tên / Full name",
            "Ngày sinh / Date of birth",
            "Giới tính / Sex",
            "Quốc tịch / Nationality",
        ]
    })

    assert result["side"] == "FRONT"
    assert result["frontScore"] > result["backScore"]


def test_back_side_is_not_reported_as_an_unclear_front_image() -> None:
    result = classify_cccd_side({
        "normalizedText": [
            "Nơi đăng ký khai sinh / Place of birth",
            "Ngày, tháng, năm cấp / Date of issue",
            "BỘ CÔNG AN / MINISTRY OF PUBLIC SECURITY",
            "IDVNM001200000001<<<<<<<<<<<<",
        ]
    })

    assert result["side"] == "BACK"
    assert "DATE_OF_ISSUE_LABEL" in result["backEvidence"]
    assert "MRZ_PREFIX" in result["backEvidence"]

