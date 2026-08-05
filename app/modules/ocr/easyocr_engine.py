from __future__ import annotations

from typing import Any

import easyocr

from app.modules.ocr.base_engine import BaseOCREngine
from app.modules.ocr.models import OCRResult, OCRTextBox
from app.modules.ocr.text_normalizer import OCRTextNormalizer


VIETNAMESE_LETTERS = (
    "AÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ"
    "EÈÉẺẼẸÊỀẾỂỄỆ"
    "IÌÍỈĨỊ"
    "OÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ"
    "UÙÚỦŨỤƯỪỨỬỮỰ"
    "YỲÝỶỸỴĐ"
)

VIETNAMESE_ALLOWLIST = (
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    f"{VIETNAMESE_LETTERS}{VIETNAMESE_LETTERS.lower()}"
    " /.,:;-'()"
)


class EasyOCREngine(BaseOCREngine):
    """EasyOCR được cấu hình riêng cho mặt trước CCCD tiếng Việt."""

    def __init__(
        self,
        languages: list[str] | None = None,
        gpu: bool = False,
    ) -> None:
        self.reader = easyocr.Reader(
            languages or ["vi", "en"],
            gpu=gpu,
        )

    def recognize(self, image_path: str) -> OCRResult:
        """Nhận dạng chữ, số và đầy đủ dấu tiếng Việt trên ảnh."""
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
                decoder="beamsearch",
                beamWidth=5,
                batch_size=1,
                workers=0,
                allowlist=VIETNAMESE_ALLOWLIST,
                min_size=5,
                text_threshold=0.55,
                low_text=0.30,
                link_threshold=0.30,
                canvas_size=2560,
                mag_ratio=1.5,
                slope_ths=0.15,
                ycenter_ths=0.50,
                height_ths=0.50,
                width_ths=0.80,
                add_margin=0.08,
                contrast_ths=0.05,
                adjust_contrast=0.70,
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

            # Normalizer chỉ sửa nhãn; phần dữ liệu tiếng Việt vẫn giữ dấu.
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
    def convert_box_to_json_safe(box: Any) -> list[list[float]]:
        """Chuyển tọa độ bounding box sang dữ liệu JSON-safe."""
        if box is None:
            return []

        converted_box: list[list[float]] = []
        for point in box:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            converted_box.append([float(point[0]), float(point[1])])

        return converted_box
