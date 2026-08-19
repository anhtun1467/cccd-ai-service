from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.core.config import settings
from app.services.ocr_pipeline import OcrPipelineService


def _full_ocr_result() -> dict:
    return {
        "structuredData": {
            "idNumber": "042096015766",
            "fullName": "ĐINH XUÂN HOÀNG",
            "dateOfBirth": "24/11/1996",
            "gender": "Nam",
            "nationality": "Việt Nam",
            "placeOfOrigin": "Xã Sơn Tây, Huyện Hương Sơn, Hà Tĩnh",
            "placeOfResidence": "Lâm Trung Thủy, Đức Thọ, Hà Tĩnh",
            "dateOfExpiry": "24/11/2036",
        },
        "normalizedText": [
            "CĂN CƯỚC CÔNG DÂN",
            "Số / No: 042096015766",
            "Họ và tên / Full name: ĐINH XUÂN HOÀNG",
            "Ngày sinh / Date of birth: 24/11/1996",
            "Giới tính / Sex: Nam Quốc tịch / Nationality: Việt Nam",
        ],
        "textBoxes": [],
        "mergedTextBoxes": [],
    }


def test_pipeline_uses_qr_and_skips_confirmed_field_ocr(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "output_dir", str(tmp_path / "outputs"))
    monkeypatch.setattr(settings, "qr_fast_path_enabled", True)
    monkeypatch.setattr(settings, "qr_skip_confirmed_field_ocr", True)

    input_path = tmp_path / "input.jpg"
    card = np.full((630, 1000, 3), 185, dtype=np.uint8)
    assert cv2.imwrite(str(input_path), card)

    pipeline = OcrPipelineService()
    monkeypatch.setattr(
        pipeline.card_detector,
        "detect_from_path",
        lambda **kwargs: {
            "cardImage": card.copy(),
            "enhancedImage": card.copy(),
            "geometry": {
                "candidateName": "perspective_contour",
                "wholeCardReliable": True,
                "geometryRotationDegrees": 0,
            },
            "cardCandidates": [],
            "resizeRatio": 1.0,
        },
    )
    monkeypatch.setattr(
        pipeline,
        "run_full_card_ocr",
        lambda image_path: _full_ocr_result(),
    )
    monkeypatch.setattr(
        pipeline,
        "ensure_upright_orientation",
        lambda **kwargs: (
            kwargs["card_image"],
            kwargs["enhanced_image"],
            kwargs["first_ocr_result"],
            {
                "contentRotationDegrees": 0,
                "orientationRetried": False,
                "selectedScore": 30.0,
            },
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "select_best_geometry_candidate",
        lambda **kwargs: (
            kwargs["card_image"],
            kwargs["enhanced_image"],
            kwargs["first_ocr_result"],
            {
                "retried": False,
                "selectedCandidate": "perspective_contour",
                "candidates": [
                    {
                        "name": "perspective_contour",
                        "score": 30.0,
                        "selected": True,
                    }
                ],
            },
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "refine_residual_skew",
        lambda **kwargs: (
            kwargs["card_image"],
            kwargs["enhanced_image"],
            kwargs["first_ocr_result"],
            {"retried": False, "candidates": []},
        ),
    )
    monkeypatch.setattr(
        pipeline.qr_decoder,
        "decode",
        lambda image: {
            "decoded": True,
            "used": True,
            "decoder": "OpenCV QRCodeDetector",
            "attemptCount": 1,
            "selectedVariant": "card_raw",
            "elapsedMs": 12.0,
            "format": "CCCD_QR_7_FIELDS",
            "fieldCount": 7,
            "providedFields": [
                "idNumber",
                "fullName",
                "dateOfBirth",
                "gender",
                "placeOfResidence",
            ],
            "structuredData": {
                "idNumber": "042096015766",
                "fullName": "ĐINH XUÂN HOÀNG",
                "dateOfBirth": "24/11/1996",
                "gender": "Nam",
                "placeOfResidence": (
                    "Phường Bến Nghé, Quận 1, Thành phố Hồ Chí Minh"
                ),
            },
            "auxiliaryData": {
                "hasOldDocumentNumber": False,
                "hasDateOfIssue": True,
                "additionalFieldCount": 0,
            },
            "errors": [],
        },
    )
    monkeypatch.setattr(
        "app.services.ocr_pipeline.check_image_quality",
        lambda *args, **kwargs: {
            "is_valid": True,
            "is_blurry": False,
            "is_too_dark": False,
            "blur_score": 300.0,
            "brightness_score": 185.0,
        },
    )
    monkeypatch.setattr(
        "app.services.ocr_pipeline.cccd_image_enhancer.estimate_blur",
        lambda image: {"blurScore": 300.0},
    )

    captured: dict = {}

    def fake_field_ocr(**kwargs) -> dict:
        captured.update(kwargs)
        skipped = set(kwargs["skip_fields"])
        return {
            "structuredData": {
                "idNumber": None,
                "fullName": None,
                "dateOfBirth": None,
                "gender": None,
                "nationality": "Việt Nam",
                "placeOfOrigin": (
                    "Xã Sơn Tây, Huyện Hương Sơn, Hà Tĩnh"
                ),
                "placeOfResidence": None,
                "dateOfExpiry": "24/11/2036",
            },
            "fieldResults": {
                field_name: {
                    "attemptCount": 0 if field_name in skipped else 1,
                    "averageConfidence": 1.0,
                    "value": None,
                }
                for field_name in pipeline.FIELD_NAMES
            },
            "portrait": None,
            "debug": {},
        }

    monkeypatch.setattr(pipeline, "run_field_ocr", fake_field_ocr)
    monkeypatch.setattr(
        pipeline,
        "save_json_response",
        lambda **kwargs: None,
    )

    response = pipeline.process_cccd_image(str(input_path))

    expected_qr_skips = {
        "idNumber",
        "fullName",
        "dateOfBirth",
        "gender",
        "placeOfResidence",
    }
    expected_skips = expected_qr_skips | {"nationality", "dateOfExpiry"}
    assert set(captured["skip_fields"]) == expected_skips
    assert captured["skip_field_sources"]["nationality"] == (
        "VALIDATED_FULL_CARD_OCR"
    )
    assert response["status"] == "OCR_SUCCESS"
    assert response["cccdData"]["fullName"] == "ĐINH XUÂN HOÀNG"
    assert response["metadata"]["dataSources"]["fullName"] == "CCCD_QR"
    assert response["metadata"]["fieldOcrAttemptCount"] == 1
    assert response["metadata"]["fieldOcrSkippedByValidatedFullCard"] == [
        "dateOfExpiry",
        "nationality",
    ]
    assert response["metadata"]["qrFastPath"]["decoded"] is True
    assert response["metadata"]["qrFastPath"]["skippedFieldOcr"] == sorted(
        expected_qr_skips
    )
    assert response["metadata"]["qrFastPath"]["conflicts"] == [
        {
            "field": "placeOfResidence",
            "ocrSource": "FULL_CARD_OCR",
            "resolution": "CCCD_QR",
            "requiresReview": False,
        }
    ]
    assert response["metadata"]["validation"][
        "qrAdvisoryDifferenceFields"
    ] == ["placeOfResidence"]
    assert response["metadata"]["imageQuality"]["decision"] == "PASSED"
    assert "structuredData" not in response["metadata"]["qrFastPath"]
