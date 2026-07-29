from __future__ import annotations

from typing import Any

import easyocr

from app.modules.ocr.base_engine import BaseOCREngine
from app.modules.ocr.models import OCRResult, OCRTextBox
from app.modules.ocr.text_normalizer import OCRTextNormalizer


class EasyOCREngine(BaseOCREngine):
    """
    OCR Engine sß╗¡ dß╗Ñng EasyOCR.

    Quy tr├¼nh:
    1. ─Éß╗ìc v─ân bß║ún tß╗½ ß║únh.
    2. Chuß║⌐n h├│a khoß║úng trß║»ng v├á c├íc nh├ún CCCD.
    3. Chuyß╗ân tß╗ìa ─æß╗Ö bounding box sang kiß╗âu JSON-safe.
    4. Trß║ú vß╗ü OCRResult.
    """

    def __init__(
        self,
        languages: list[str] | None = None,
        gpu: bool = False,
    ) -> None:
        self.reader = easyocr.Reader(
            languages or ["en"],
            gpu=gpu,
        )

    def recognize(self, image_path: str) -> OCRResult:
        """
        Nhß║¡n dß║íng v─ân bß║ún trong ß║únh.

        Args:
            image_path: ─É╞░ß╗¥ng dß║½n ─æß║┐n ß║únh cß║ºn OCR.

        Returns:
            OCRResult chß╗⌐a danh s├ích v─ân bß║ún v├á c├íc bounding box.
        """

        if not image_path or not image_path.strip():
            return OCRResult(
                success=False,
                message="─É╞░ß╗¥ng dß║½n ß║únh kh├┤ng hß╗úp lß╗ç",
                raw_text=[],
                text_boxes=[],
            )

        try:
            results = self.reader.readtext(
                image_path,
                detail=1,
                paragraph=False,
            )
        except Exception as error:
            return OCRResult(
                success=False,
                message=f"EasyOCR thß║Ñt bß║íi: {error}",
                raw_text=[],
                text_boxes=[],
            )

        raw_text: list[str] = []
        text_boxes: list[OCRTextBox] = []

        for item in results:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue

            box = item[0]
            original_text = str(item[1]).strip()
            confidence = float(item[2])

            if not original_text:
                continue

            normalized_text = OCRTextNormalizer.normalize(original_text)

            if not normalized_text:
                continue

            raw_text.append(normalized_text)

            text_boxes.append(
                OCRTextBox(
                    text=normalized_text,
                    confidence=confidence,
                    box=self.convert_box_to_json_safe(box),
                )
            )

        if not text_boxes:
            return OCRResult(
                success=False,
                message="Kh├┤ng ph├ít hiß╗çn ─æ╞░ß╗úc v─ân bß║ún trong ß║únh",
                raw_text=[],
                text_boxes=[],
            )

        return OCRResult(
            success=True,
            message="OCR th├ánh c├┤ng",
            raw_text=raw_text,
            text_boxes=text_boxes,
        )

    @staticmethod
    def convert_box_to_json_safe(
        box: Any,
    ) -> list[list[float]]:
        """
        Chuyß╗ân tß╗ìa ─æß╗Ö bounding box cß╗ºa EasyOCR sang danh s├ích float
        ─æß╗â c├│ thß╗â serialize th├ánh JSON.
        """

        if box is None:
            return []

        converted_box: list[list[float]] = []

        for point in box:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue

            converted_box.append(
                [
                    float(point[0]),
                    float(point[1]),
                ]
            )

        return converted_box
