from __future__ import annotations

from pathlib import Path
from typing import Any

from app.modules.ocr.base_engine import BaseOCREngine
from app.modules.ocr.easyocr_engine import EasyOCREngine
from app.modules.ocr.field_cropper import CCCDFieldCropper
from app.modules.ocr.text_normalizer import OCRTextNormalizer


class FieldOCRService:
    """
    OCR từng vùng thông tin CCCD.

    Quy trình:
    Card image
        -> Field Cropper
        -> OCR từng field
        -> Normalize text
        -> Structured data
    """

    OCR_FIELDS = (
        "idNumber",
        "fullName",
        "dateOfBirth",
        "gender",
        "nationality",
        "placeOfOrigin",
        "placeOfResidence",
        "dateOfExpiry",
    )

    def __init__(
        self,
        engine: BaseOCREngine | None = None,
    ) -> None:
        self.engine = engine or EasyOCREngine()
        self.cropper = CCCDFieldCropper()

    def extract_fields(
        self,
        card_image_path: str,
        output_dir: str,
    ) -> dict[str, Any]:
        """
        Cắt và OCR từng vùng trên CCCD.
        """

        field_results = self.cropper.crop_fields_from_path(
            image_path=card_image_path,
            output_dir=output_dir,
        )

        structured_data: dict[str, str | None] = {}
        field_ocr_results: dict[str, Any] = {}

        for field_name in self.OCR_FIELDS:
            field_result = field_results.get(field_name)

            if not field_result:
                structured_data[field_name] = None
                continue

            image_path = field_result.get("imagePath")

            if not image_path:
                structured_data[field_name] = None
                continue

            ocr_result = self.engine.recognize(
                str(image_path)
            )

            normalized_lines = OCRTextNormalizer.normalize_lines(
                ocr_result.raw_text
            )

            normalized_text = self.join_field_text(
                field_name=field_name,
                lines=normalized_lines,
            )

            cleaned_value = self.clean_field_value(
                field_name=field_name,
                value=normalized_text,
            )

            structured_data[field_name] = cleaned_value

            field_ocr_results[field_name] = {
                "success": ocr_result.success,
                "message": ocr_result.message,
                "rawText": ocr_result.raw_text,
                "normalizedText": normalized_lines,
                "value": cleaned_value,
                "imagePath": image_path,
                "rawImagePath": field_result.get(
                    "rawImagePath"
                ),
                "averageConfidence": (
                    self.calculate_average_confidence(
                        ocr_result.text_boxes
                    )
                ),
            }

        return {
            "structuredData": structured_data,
            "fieldResults": field_ocr_results,
            "portrait": field_results.get("portrait"),
            "debug": field_results.get("_debug"),
        }

    @staticmethod
    def join_field_text(
        field_name: str,
        lines: list[str],
    ) -> str:
        """
        Ghép các dòng OCR của từng field.
        """

        if not lines:
            return ""

        if field_name in {
            "placeOfOrigin",
            "placeOfResidence",
        }:
            return ", ".join(lines)

        return " ".join(lines)

    @staticmethod
    def clean_field_value(
        field_name: str,
        value: str,
    ) -> str | None:
        """
        Làm sạch giá trị OCR theo từng trường.
        """

        if not value:
            return None

        value = value.strip()

        if field_name == "idNumber":
            digits = "".join(
                character
                for character in value
                if character.isdigit()
            )

            return digits if len(digits) == 12 else None

        if field_name in {
            "dateOfBirth",
            "dateOfExpiry",
        }:
            value = value.replace("-", "/")
            value = value.replace(".", "/")

        if field_name == "fullName":
            value = " ".join(value.split())
            value = value.upper()

        if field_name == "gender":
            lowered = value.lower()

            if "nam" in lowered or "male" in lowered:
                return "Nam"

            if "nữ" in lowered or "nu" in lowered:
                return "Nữ"

        if field_name == "nationality":
            lowered = value.lower()

            if (
                "viet nam" in lowered
                or "vietnam" in lowered
                or "vict nam" in lowered
            ):
                return "Viet Nam"

        return value or None

    @staticmethod
    def calculate_average_confidence(
        text_boxes: list[Any],
    ) -> float:
        """
        Tính confidence trung bình của field.
        """

        if not text_boxes:
            return 0.0

        scores: list[float] = []

        for item in text_boxes:
            confidence = getattr(
                item,
                "confidence",
                0.0,
            )

            try:
                scores.append(float(confidence))
            except (TypeError, ValueError):
                continue

        if not scores:
            return 0.0

        return round(
            sum(scores) / len(scores),
            4,
        )


field_ocr_service = FieldOCRService()