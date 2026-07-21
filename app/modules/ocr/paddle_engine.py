from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("ONEDNN_VERBOSE", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from paddleocr import PaddleOCR

from app.modules.ocr.base_engine import BaseOCREngine
from app.modules.ocr.models import OCRResult, OCRTextBox


class PaddleOCREngine(BaseOCREngine):
    """
    OCR Engine sử dụng PaddleOCR.

    Nhiệm vụ:
    - Nhận đường dẫn ảnh.
    - Chạy PaddleOCR.
    - Chuẩn hóa kết quả về OCRResult.
    """

    def __init__(self) -> None:
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="en",
        )

    def recognize(self, image_path: str) -> OCRResult:
        pages = self.ocr.predict(image_path)

        raw_text: list[str] = []
        text_boxes: list[OCRTextBox] = []

        for page in pages:
            page_data = self._normalize_page(page)

            rec_texts = page_data.get("rec_texts", [])
            rec_scores = page_data.get("rec_scores", [])
            rec_boxes = page_data.get("rec_boxes", [])

            for text, score, box in zip(rec_texts, rec_scores, rec_boxes):
                clean_text = str(text).strip()

                if not clean_text:
                    continue

                raw_text.append(clean_text)

                text_boxes.append(
                    OCRTextBox(
                        text=clean_text,
                        confidence=float(score),
                        box=self._box_to_list(box),
                    )
                )

        return OCRResult(
            success=True,
            message="OCR thành công",
            raw_text=raw_text,
            text_boxes=text_boxes,
        )

    def _normalize_page(self, page: Any) -> dict:
        if isinstance(page, dict):
            return page

        if hasattr(page, "json"):
            return page.json

        if hasattr(page, "__dict__"):
            return page.__dict__

        return {}

    def _box_to_list(self, box: Any) -> list:
        if hasattr(box, "tolist"):
            return box.tolist()

        if isinstance(box, list):
            return box

        return []