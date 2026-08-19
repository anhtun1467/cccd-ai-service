from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.modules.ocr.easyocr_engine import EasyOCREngine
from app.services.ocr_pipeline import OcrPipelineService


class _RecordingReader:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def readtext(self, image_path: str, **kwargs):
        self.calls.append({"imagePath": image_path, **kwargs})
        return []


def test_processed_field_is_not_magnified_twice() -> None:
    engine = EasyOCREngine.__new__(EasyOCREngine)
    engine.reader = _RecordingReader()

    engine._recognize(
        image_path="fullName_value.jpg",
        allowlist="ABC",
        field_mode=True,
    )
    engine._recognize(
        image_path="fullName_value_raw.jpg",
        allowlist="ABC",
        field_mode=True,
    )

    processed, raw = engine.reader.calls
    assert processed["canvas_size"] == 2048
    assert processed["mag_ratio"] == 1.0
    assert raw["mag_ratio"] == 1.6
    assert processed["batch_size"] == raw["batch_size"] == 4


def test_full_card_uses_bounded_canvas() -> None:
    engine = EasyOCREngine.__new__(EasyOCREngine)
    engine.reader = _RecordingReader()

    engine._recognize(
        image_path="enhanced_card.jpg",
        allowlist="ABC",
        field_mode=False,
    )

    call = engine.reader.calls[0]
    assert call["canvas_size"] == 2304
    assert call["mag_ratio"] == 1.35


def test_validated_full_card_skips_only_strict_fields() -> None:
    pipeline = OcrPipelineService()

    skipped = pipeline.select_validated_full_card_field_ocr_skips(
        {
            "idNumber": "012345678901",
            "fullName": "NGUYỄN VĂN A",
            "dateOfBirth": "01/01/2000",
            "gender": "Nam",
            "nationality": "Việt Nam",
            "placeOfOrigin": "Phường A, Thành phố B",
            "placeOfResidence": "Phường C, Thành phố D",
            "dateOfExpiry": "01/01/2030",
        },
        data_sources={
            "placeOfOrigin": "SPATIAL_OCR",
            "placeOfResidence": "FULL_CARD_OCR",
        },
    )

    assert skipped == {
        "idNumber",
        "dateOfBirth",
        "gender",
        "nationality",
        "placeOfOrigin",
        "dateOfExpiry",
    }


def _weak_ocr_result() -> dict:
    return {
        "structuredData": {},
        "normalizedText": [],
        "textBoxes": [],
    }


def _strong_ocr_result() -> dict:
    return {
        "structuredData": {
            "idNumber": "042096015766",
            "fullName": "ĐINH XUÂN HOÀNG",
            "dateOfBirth": "24/11/1996",
            "gender": "Nam",
            "nationality": "Việt Nam",
        },
        "normalizedText": [
            "CĂN CƯỚC CÔNG DÂN",
            "Số / No: 042096015766",
            "Họ và tên / Full name: ĐINH XUÂN HOÀNG",
            "Ngày sinh / Date of birth: 24/11/1996",
            "Giới tính / Sex: Nam Quốc tịch / Nationality: Việt Nam",
            "Quê quán / Place of origin",
            "Nơi thường trú / Place of residence",
            "Có giá trị đến / Date of expiry",
        ],
        "textBoxes": [],
    }


def test_reliable_primary_geometry_does_not_run_candidate_ocr(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pipeline = OcrPipelineService()
    card = np.full((60, 96, 3), 180, dtype=np.uint8)
    detection_result = {
        "geometry": {
            "candidateName": "perspective_contour",
            "wholeCardReliable": True,
            "sourceCoverageRatio": 0.85,
            "perspectiveSeverity": 0.05,
        },
        "cardCandidates": [{"name": "unused", "cardImage": card.copy()}],
    }

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Không được OCR ứng viên khi ảnh chính đã tin cậy")

    monkeypatch.setattr(pipeline, "run_full_card_ocr", fail_if_called)
    debug_dir = tmp_path / "debug"
    debug_dir.mkdir()
    _, _, _, selection = pipeline.select_best_geometry_candidate(
        card_image=card,
        enhanced_image=card.copy(),
        first_ocr_result=_weak_ocr_result(),
        detection_result=detection_result,
        card_output_path=tmp_path / "card.jpg",
        enhanced_output_path=tmp_path / "enhanced.jpg",
        debug_dir=debug_dir,
    )

    assert selection["retried"] is False
    assert selection["retryReason"] == "PRIMARY_GEOMETRY_ACCEPTED"


def test_only_verified_hough_candidate_is_ocrd_in_upright_direction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pipeline = OcrPipelineService()
    primary = np.full((60, 96, 3), 40, dtype=np.uint8)
    candidate = np.full((60, 96, 3), 120, dtype=np.uint8)
    candidate[0:12, 0:12] = (0, 0, 255)
    ignored = np.full((60, 96, 3), 220, dtype=np.uint8)
    detection_result = {
        "geometry": {
            "candidateName": "perspective_contour",
            "wholeCardReliable": False,
            "sourceCoverageRatio": 0.50,
            "perspectiveSeverity": 0.20,
        },
        "cardCandidates": [
            {
                "name": "hough_whole_card_1",
                "cardImage": candidate,
                "geometry": {
                    "wholeCardReliable": True,
                    "detectionMetrics": {"wholeCardCandidate": True},
                },
            },
            {"name": "must_not_run", "cardImage": ignored, "geometry": {}},
        ],
    }
    calls: list[np.ndarray] = []

    def recognize(path: Path) -> dict:
        image = cv2.imread(str(path))
        assert image is not None
        calls.append(image)
        return _strong_ocr_result()

    monkeypatch.setattr(pipeline, "run_full_card_ocr", recognize)
    monkeypatch.setattr(
        pipeline.card_detector.enhancer,
        "enhance",
        lambda image: {"final": image.copy()},
    )
    debug_dir = tmp_path / "debug"
    debug_dir.mkdir()
    selected_card, _, _, selection = (
        pipeline.select_best_geometry_candidate(
            card_image=primary,
            enhanced_image=primary.copy(),
            first_ocr_result=_weak_ocr_result(),
            detection_result=detection_result,
            card_output_path=tmp_path / "card.jpg",
            enhanced_output_path=tmp_path / "enhanced.jpg",
            debug_dir=debug_dir,
            content_rotation_degrees=180,
        )
    )

    assert len(calls) == 1
    assert selection["selectedCandidate"] == "hough_whole_card_1"
    assert np.array_equal(selected_card, cv2.rotate(candidate, cv2.ROTATE_180))
    # Mảng đã xoay: ô đỏ từ góc trên-trái phải nằm ở góc dưới-phải.
    assert float(np.mean(calls[0][-10:, -10:, 2])) > 200.0


def test_valid_qr_and_strong_ocr_skip_expensive_skew_ocr_trials(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pipeline = OcrPipelineService()
    card = np.full((630, 1000, 3), 190, dtype=np.uint8)
    debug_dir = tmp_path / "debug"
    debug_dir.mkdir()
    monkeypatch.setattr(
        pipeline.geometry_refiner,
        "estimate_text_skew",
        lambda *args, **kwargs: {
            "angleDegrees": 2.4,
            "confidence": 0.8,
            "lineCount": 12,
            "reliable": True,
            "source": "hough",
        },
    )
    monkeypatch.setattr(
        pipeline.geometry_refiner,
        "build_correction_angles",
        lambda estimate: [-2.4, -1.2],
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Không được chạy lại full-card OCR")

    monkeypatch.setattr(pipeline, "run_full_card_ocr", fail_if_called)

    _, _, result, info = pipeline.refine_residual_skew(
        card_image=card,
        enhanced_image=card.copy(),
        first_ocr_result=_strong_ocr_result(),
        card_output_path=tmp_path / "card.jpg",
        enhanced_output_path=tmp_path / "enhanced.jpg",
        debug_dir=debug_dir,
        qr_decoded=True,
    )

    assert result == _strong_ocr_result()
    assert info["retried"] is False
    assert info["retrySkipped"] is True
    assert info["skipReason"] == "QR_AND_STRONG_OCR"
    assert info["candidates"] == []


def test_soft_blur_warning_does_not_raise_parser_field_requirement() -> None:
    pipeline = OcrPipelineService()
    result = pipeline.evaluate_ocr_readability(
        merged_data={
            "idNumber": "012345678901",
            "fullName": "NGUYỄN VĂN A",
        },
        cropped_quality={
            "is_blurry": True,
            "is_too_dark": False,
            "blur_score": 62.0,
            "brightness_score": 155.0,
        },
        raw_text=["CĂN CƯỚC", "012345678901", "NGUYỄN VĂN A"],
    )

    assert result["hardLowQuality"] is False
    assert result["minimumReadableCoreFields"] == 2
    assert result["isReadable"] is True


def test_hard_blur_still_requires_three_core_fields() -> None:
    pipeline = OcrPipelineService()
    result = pipeline.evaluate_ocr_readability(
        merged_data={
            "idNumber": "012345678901",
            "fullName": "NGUYỄN VĂN A",
        },
        cropped_quality={
            "is_blurry": True,
            "is_too_dark": False,
            "blur_score": 12.0,
            "brightness_score": 155.0,
        },
        raw_text=["CĂN CƯỚC", "012345678901", "NGUYỄN VĂN A"],
    )

    assert result["hardLowQuality"] is True
    assert result["minimumReadableCoreFields"] == 3
    assert result["isReadable"] is False
