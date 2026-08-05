from __future__ import annotations

from typing import Any

from app.modules.ocr.base_engine import BaseOCREngine
from app.modules.ocr.easyocr_engine import EasyOCREngine
from app.modules.ocr.line_merger import OCRLineMerger
from app.modules.ocr.models import OCRResult
from app.modules.ocr.parser.regex_parser import CCCDRegexParser
from app.modules.ocr.text_normalizer import OCRTextNormalizer
from app.modules.ocr.validator import CCCDValidator


class OCRService:
    """
    Service điều phối pipeline OCR CCCD:

    OCR Engine
        -> Text Normalizer
        -> Line Merger
        -> Regex Parser
        -> Validator
    """

    def __init__(
        self,
        engine: BaseOCREngine | None = None,
    ) -> None:
        self.engine = engine or EasyOCREngine()

        self.line_merger = OCRLineMerger(
            # Ngưỡng cũ 0.6/3.5 làm các box ở hai cột cuối thẻ
            # (hạn sử dụng bên trái, nơi thường trú bên phải) bị nối
            # thành một dòng. Ngưỡng chặt hơn vẫn ghép được các cụm
            # cùng dòng nhưng giữ hai vùng này độc lập.
            vertical_tolerance_ratio=0.25,
            maximum_horizontal_gap_ratio=1.8,
        )

        self.parser = CCCDRegexParser()
        self.validator = CCCDValidator()

    def recognize(
        self,
        image_path: str,
    ) -> OCRResult:
        """
        Thực hiện OCR ảnh đầu vào.
        """

        return self.engine.recognize(image_path)

    def extract_cccd_info(
        self,
        image_path: str,
    ) -> dict[str, Any]:
        """
        Đọc và trích xuất thông tin CCCD từ ảnh.

        Args:
            image_path: Đường dẫn ảnh CCCD đã được xử lý.

        Returns:
            Kết quả OCR, dữ liệu CCCD, validation và text box.
        """

        ocr_result = self.recognize(image_path)

        if not ocr_result.success:
            return {
                "ocrSuccess": False,
                "ocrMessage": ocr_result.message,
                "structuredData": {},
                "validation": {
                    "isValid": False,
                    "errors": [ocr_result.message],
                },
                "normalizedText": [],
                "textBoxes": [],
                "mergedTextBoxes": [],
            }

        normalized_text_boxes = self.normalize_text_boxes(
            ocr_result
        )

        merged_text_boxes = self.line_merger.merge(
            normalized_text_boxes
        )

        normalized_lines = OCRTextNormalizer.normalize_lines(
            [
                str(item.get("text", ""))
                for item in merged_text_boxes
            ]
        )

        # Nếu line merger không ghép được, dùng dữ liệu OCR đã chuẩn hóa.
        if not normalized_lines:
            normalized_lines = OCRTextNormalizer.normalize_lines(
                ocr_result.raw_text
            )

        parsed_data = self.parser.parse(
            normalized_lines
        )

        validation_result = self.validator.validate(
            parsed_data
        )

        return {
            "ocrSuccess": True,
            "ocrMessage": ocr_result.message,
            "structuredData": parsed_data,
            "validation": validation_result,
            "normalizedText": normalized_lines,
            "textBoxes": normalized_text_boxes,
            "mergedTextBoxes": merged_text_boxes,
        }

    @staticmethod
    def normalize_text_boxes(
        ocr_result: OCRResult,
    ) -> list[dict[str, Any]]:
        """
        Chuyển OCRTextBox sang dictionary và chuẩn hóa nội dung.
        """

        normalized_boxes: list[dict[str, Any]] = []

        for item in ocr_result.text_boxes:
            original_text = str(item.text).strip()

            if not original_text:
                continue

            normalized_text = OCRTextNormalizer.normalize(
                original_text
            )

            if not normalized_text:
                continue

            normalized_boxes.append(
                {
                    "text": normalized_text,
                    "originalText": original_text,
                    "confidence": float(item.confidence),
                    "box": item.box,
                }
            )

        return normalized_boxes


ocr_service = OCRService()
