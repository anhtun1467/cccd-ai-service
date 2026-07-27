from __future__ import annotations

from typing import Any

import easyocr

from app.modules.ocr.base_engine import BaseOCREngine
from app.modules.ocr.models import OCRResult, OCRTextBox
from app.modules.ocr.text_normalizer import OCRTextNormalizer


class EasyOCREngine(BaseOCREngine):
    """
    OCR Engine sử dụng EasyOCR.

    Quy trình:
    1. Đọc văn bản từ ảnh.
    2. Chuẩn hóa khoảng trắng và các nhãn CCCD.
    3. Chuyển tọa độ bounding box sang kiểu JSON-safe.
    4. Trả về OCRResult.
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
        Nhận dạng văn bản trong ảnh.

        Args:
            image_path: Đường dẫn đến ảnh cần OCR.

        Returns:
            OCRResult chứa danh sách văn bản và các bounding box.
        """

        if not image_path or not image_path.strip():
            return OCRResult(
                success=False,
                message="Đường dẫn ảnh không hợp lệ",
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
                message=f"EasyOCR thất bại: {error}",
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
                message="Không phát hiện được văn bản trong ảnh",
                raw_text=[],
                text_boxes=[],
            )

        return OCRResult(
            success=True,
            message="OCR thành công",
            raw_text=raw_text,
            text_boxes=text_boxes,
        )

    @staticmethod
    def convert_box_to_json_safe(
        box: Any,
    ) -> list[list[float]]:
        """
        Chuyển tọa độ bounding box của EasyOCR sang danh sách float
        để có thể serialize thành JSON.
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
