from __future__ import annotations

import os

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
from paddleocr import PaddleOCR

from app.modules.ocr.base_engine import BaseOCREngine
from app.modules.ocr.models import OCRResult, OCRTextBox


class PaddleOCREngine(BaseOCREngine):

    def __init__(self) -> None:

        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="en",
        )

    def recognize(
        self,
        image_path: str,
    ) -> OCRResult:

        result = self.ocr.predict(image_path)

        raw_text: list[str] = []

        text_boxes: list[OCRTextBox] = []

        for page in result:

            rec_texts = page["rec_texts"]

            rec_scores = page["rec_scores"]

            rec_boxes = page["rec_boxes"]

            for text, score, box in zip(
                rec_texts,
                rec_scores,
                rec_boxes,
            ):

                raw_text.append(text)

                text_boxes.append(
                    OCRTextBox(
                        text=text,
                        confidence=float(score),
                        box=box.tolist(),
                    )
                )

        return OCRResult(
            success=True,
            message="OCR thành công",
            raw_text=raw_text,
            text_boxes=text_boxes,
        )