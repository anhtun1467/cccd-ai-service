from __future__ import annotations

from app.modules.ocr.base_engine import BaseOCREngine
from app.modules.ocr.easyocr_engine import EasyOCREngine
from app.modules.ocr.models import OCRResult
from app.modules.ocr.regex_parser import CCCDRegexParser
from app.modules.ocr.text_normalizer import TextNormalizer
from app.modules.ocr.validator import CCCDValidator


class OCRService:
    """
    Service điều phối OCR:
    OCR Engine -> Text Normalizer -> Regex Parser -> Validator.
    """

    def __init__(self, engine: BaseOCREngine | None = None) -> None:
        self.engine = engine or EasyOCREngine()
        self.normalizer = TextNormalizer()
        self.parser = CCCDRegexParser()
        self.validator = CCCDValidator()

    def recognize(self, image_path: str) -> OCRResult:
        return self.engine.recognize(image_path)

    def extract_cccd_info(self, image_path: str) -> dict:
        ocr_result = self.recognize(image_path)

        normalized_text = self.normalizer.normalize(
            ocr_result.raw_text
        )

        parsed_data = self.parser.parse(
            normalized_text
        )

        validation_result = self.validator.validate(
            parsed_data
        )

        return {
            "ocrSuccess": ocr_result.success,
            "ocrMessage": ocr_result.message,
            "structuredData": parsed_data,
            "validation": validation_result,
            "normalizedText": normalized_text,
            "textBoxes": [
                {
                    "text": item.text,
                    "confidence": item.confidence,
                    "box": item.box,
                }
                for item in ocr_result.text_boxes
            ],
        }


ocr_service = OCRService()