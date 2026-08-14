from __future__ import annotations

from pathlib import Path

from app.modules.ocr.field_ocr_service import FieldOCRService
from app.modules.ocr.models import OCRResult, OCRTextBox


class FakeEngine:
    def recognize(self, image_path: str) -> OCRResult:
        return self.recognize_field(image_path, "fullName")

    def recognize_field(
        self,
        image_path: str,
        field_name: str,
    ) -> OCRResult:
        return OCRResult(
            success=True,
            message="ok",
            raw_text=["BUI THI DUYEN"],
            text_boxes=[
                OCRTextBox(
                    text="BUI THI DUYEN",
                    confidence=0.92,
                    box=[[0, 0], [300, 0], [300, 60], [0, 60]],
                )
            ],
        )


class FakeGlyphMatcher:
    def refine_ocr_boxes(
        self,
        image_path: str | Path,
        text_boxes: list[OCRTextBox],
        field_name: str,
    ) -> tuple[dict[int, str], dict]:
        return (
            {0: "BÙI THỊ DUYÊN"},
            {
                "enabled": True,
                "available": True,
                "fieldName": field_name,
                "averageBestScore": 0.94,
                "coverage": 1.0,
                "applied": True,
                "corrections": [
                    {"from": "U", "to": "Ù", "score": 0.95}
                ],
            },
        )


class FailingGlyphMatcher:
    def refine_ocr_boxes(self, *args, **kwargs):
        raise RuntimeError("atlas test error")


def test_refined_glyph_text_is_used_before_field_cleaning() -> None:
    service = FieldOCRService(
        engine=FakeEngine(),
        glyph_matcher=FakeGlyphMatcher(),
    )

    candidate = service.recognize_field_candidate(
        field_name="fullName",
        image_path=Path("synthetic-field.jpg"),
        variant="value_processed",
    )

    assert candidate["value"] == "BÙI THỊ DUYÊN"
    assert candidate["glyphRefinedText"] == ["BÙI THỊ DUYÊN"]
    assert candidate["glyphMatch"]["applied"] is True
    assert candidate["_glyphScore"] == 0.94


def test_glyph_failure_keeps_original_easyocr_result() -> None:
    service = FieldOCRService(
        engine=FakeEngine(),
        glyph_matcher=FailingGlyphMatcher(),
    )

    candidate = service.recognize_field_candidate(
        field_name="fullName",
        image_path=Path("synthetic-field.jpg"),
        variant="value_processed",
    )

    assert candidate["value"] == "BUI THI DUYEN"
    assert candidate["glyphMatch"]["skippedReason"] == "MATCHER_ERROR"
    assert candidate["_glyphScore"] == 0.0
