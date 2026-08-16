from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import easyocr

from app.modules.ocr.base_engine import BaseOCREngine
from app.modules.ocr.models import OCRResult, OCRTextBox
from app.modules.ocr.text_normalizer import OCRTextNormalizer
from app.modules.ocr.vietnamese_charset import (
    FIELD_ALLOWLISTS,
    NUMERIC_OCR_ALLOWLIST,
    VIETNAMESE_LETTERS_UPPER,
    VIETNAMESE_OCR_ALLOWLIST,
)


# Giữ các tên cũ để tương thích code/test hiện tại. Nguồn duy nhất của bảng
# ký tự nằm trong vietnamese_charset.py, tránh thiếu dấu giữa các engine.
VIETNAMESE_LETTERS = VIETNAMESE_LETTERS_UPPER
VIETNAMESE_ALLOWLIST = VIETNAMESE_OCR_ALLOWLIST
NUMERIC_ALLOWLIST = NUMERIC_OCR_ALLOWLIST


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
        return self._recognize(
            image_path=image_path,
            allowlist=VIETNAMESE_ALLOWLIST,
            field_mode=False,
        )

    def recognize_field(
        self,
        image_path: str,
        field_name: str,
    ) -> OCRResult:
        """OCR một vùng field với bảng ký tự và ngưỡng dành cho ảnh mờ."""
        return self._recognize(
            image_path=image_path,
            allowlist=FIELD_ALLOWLISTS.get(
                field_name,
                VIETNAMESE_ALLOWLIST,
            ),
            field_mode=True,
        )

    def _recognize(
        self,
        image_path: str,
        allowlist: str,
        field_mode: bool,
    ) -> OCRResult:
        if not image_path or not image_path.strip():
            return OCRResult(
                success=False,
                message="Đường dẫn ảnh không hợp lệ",
                raw_text=[],
                text_boxes=[],
            )

        try:
            # Ảnh processed đã được cropper phóng 2.7-3.5 lần. Phóng tiếp
            # 1.8 lần trong EasyOCR làm detector xử lý ảnh rộng tới 3200 px
            # ở mỗi retry mà không tạo thêm chi tiết. Ảnh raw chưa phóng vẫn
            # giữ mag_ratio cao hơn để bảo toàn khả năng đọc chữ nhỏ.
            raw_field_variant = bool(
                field_mode
                and "_raw" in Path(image_path).stem.casefold()
            )
            field_mag_ratio = 1.60 if raw_field_variant else 1.00
            # Torch mới cảnh báo pin_memory ở mỗi lần EasyOCR chạy CPU.
            # Đây không phải lỗi và gây ngập log khi OCR nhiều field.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=(
                        ".*pin_memory.*no accelerator.*"
                    ),
                    category=UserWarning,
                )
                results = self.reader.readtext(
                    image_path,
                    detail=1,
                    paragraph=False,
                    # Beam search của EasyOCR có thể phát sinh overflow
                    # trên các dòng dài. Greedy ổn định hơn và các nguồn
                    # ảnh khác nhau được hợp nhất ở tầng field/fuser.
                    decoder="greedy",
                    batch_size=4,
                    workers=0,
                    allowlist=allowlist,
                    min_size=3 if field_mode else 5,
                    text_threshold=0.35 if field_mode else 0.55,
                    low_text=0.15 if field_mode else 0.30,
                    link_threshold=0.20 if field_mode else 0.30,
                    canvas_size=2048 if field_mode else 2560,
                    mag_ratio=field_mag_ratio if field_mode else 1.5,
                    slope_ths=0.20 if field_mode else 0.15,
                    ycenter_ths=0.55 if field_mode else 0.50,
                    height_ths=0.60 if field_mode else 0.50,
                    width_ths=1.00 if field_mode else 0.80,
                    add_margin=0.12 if field_mode else 0.08,
                    contrast_ths=0.10 if field_mode else 0.05,
                    adjust_contrast=0.65 if field_mode else 0.70,
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
