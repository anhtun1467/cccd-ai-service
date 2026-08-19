from __future__ import annotations

from pathlib import Path

from app.modules.ocr.field_ocr_service import FieldOCRService


def test_qr_fields_do_not_call_ocr_engine(
    tmp_path: Path,
    monkeypatch,
) -> None:
    card_path = tmp_path / "card.jpg"
    card_path.write_bytes(b"test-placeholder")
    service = FieldOCRService(engine=object())

    crop_results = {
        field_name: {
            "imagePath": str(tmp_path / f"{field_name}.jpg"),
            "rawImagePath": str(tmp_path / f"{field_name}_raw.jpg"),
            "box": [[0, 0], [1, 0], [1, 1], [0, 1]],
        }
        for field_name in service.OCR_FIELDS
    }
    crop_results["portrait"] = None
    crop_results["_debug"] = {}
    monkeypatch.setattr(
        service.cropper,
        "crop_fields_from_path",
        lambda **kwargs: crop_results,
    )
    monkeypatch.setattr(
        service,
        "prepare_portrait_result",
        lambda value: None,
    )
    monkeypatch.setattr(
        service,
        "recognize_field_candidate",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("OCR engine không được gọi")
        ),
    )

    result = service.extract_fields(
        card_image_path=str(card_path),
        output_dir=str(tmp_path / "fields"),
        skip_fields=service.OCR_FIELDS,
    )

    assert result["debug"]["skippedOcrFields"] == sorted(
        service.OCR_FIELDS
    )
    for field_name in service.OCR_FIELDS:
        field_result = result["fieldResults"][field_name]
        assert field_result["attemptCount"] == 0
        assert field_result["skipped"] is True
        assert field_result["skipReason"] == "CCCD_QR_FAST_PATH"


def test_validated_full_card_skip_has_correct_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    card_path = tmp_path / "card.jpg"
    card_path.write_bytes(b"test-placeholder")
    service = FieldOCRService(engine=object())
    crop_results = {
        field_name: {"imagePath": str(tmp_path / f"{field_name}.jpg")}
        for field_name in service.OCR_FIELDS
    }
    crop_results["portrait"] = None
    crop_results["_debug"] = {}
    monkeypatch.setattr(
        service.cropper,
        "crop_fields_from_path",
        lambda **kwargs: crop_results,
    )
    monkeypatch.setattr(service, "prepare_portrait_result", lambda value: None)
    monkeypatch.setattr(
        service,
        "recognize_field_candidate",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("OCR engine không được gọi")
        ),
    )

    result = service.extract_fields(
        card_image_path=str(card_path),
        output_dir=str(tmp_path / "fields"),
        skip_fields=service.OCR_FIELDS,
        skip_field_sources={
            field_name: "VALIDATED_FULL_CARD_OCR"
            for field_name in service.OCR_FIELDS
        },
    )

    for field_result in result["fieldResults"].values():
        assert field_result["skipReason"] == "VALIDATED_FULL_CARD_OCR"
        assert field_result["fullCardValidated"] is True
        assert field_result["qrValidated"] is False
