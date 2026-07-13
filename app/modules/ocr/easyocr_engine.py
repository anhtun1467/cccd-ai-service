from __future__ import annotations

import easyocr

from app.modules.ocr.base_engine import BaseOCREngine
from app.modules.ocr.models import OCRResult, OCRTextBox


class EasyOCREngine(BaseOCREngine):
    """
    OCR Engine sử dụng EasyOCR.
    """

    def __init__(self) -> None:
        self.reader = easyocr.Reader(
            ["en"],
            gpu=False,
        )

    def recognize(self, image_path: str) -> OCRResult:
        result = self.reader.readtext(image_path)

        raw_text: list[str] = []
        text_boxes: list[OCRTextBox] = []

        for item in result:
            box = item[0]
            text = str(item[1]).strip()
            confidence = float(item[2])

            if not text:
                continue

            raw_text.append(text)

            text_boxes.append(
                OCRTextBox(
                    text=text,
                    confidence=confidence,
                    box=self.convert_box_to_json_safe(box),
                )
            )

        return OCRResult(
            success=True,
            message="OCR thành công",
            raw_text=raw_text,
            text_boxes=text_boxes,
        )

    def convert_box_to_json_safe(self, box) -> list[list[float]]:
        return [
            [float(point[0]), float(point[1])]
            for point in box
        ]