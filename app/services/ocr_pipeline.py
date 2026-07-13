from pathlib import Path
from typing import Any
import time

import cv2

from app.core.config import settings
from app.modules.card_detection.detector import CardDetector
from app.modules.ocr.service import ocr_service


class OcrPipelineService:
    """
    Điều phối pipeline OCR CCCD:
    Upload Image -> Card Detection -> Enhancement -> OCR -> Parser -> Validator.
    """

    def __init__(self) -> None:
        self.card_detector = CardDetector()
        self.ocr_service = ocr_service

    def process_cccd_image(self, image_path: str) -> dict[str, Any]:
        start_time = time.perf_counter()

        image_file = Path(image_path)
        file_stem = image_file.stem

        output_dir = Path(settings.output_dir)
        debug_dir = Path("storage/debug") / file_stem

        output_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)

        detection_result = self.card_detector.detect_from_path(
            image_path=image_path,
            output_dir=str(debug_dir),
        )

        card_output_path = output_dir / f"{file_stem}_card.jpg"
        enhanced_output_path = output_dir / f"{file_stem}_enhanced.jpg"

        cv2.imwrite(str(card_output_path), detection_result["cardImage"])
        cv2.imwrite(str(enhanced_output_path), detection_result["enhancedImage"])

        ocr_result = self.ocr_service.extract_cccd_info(
            str(enhanced_output_path)
        )

        structured_data = self.make_json_safe(ocr_result["structuredData"])
        text_boxes = self.make_json_safe(ocr_result["textBoxes"])

        processing_time = round(time.perf_counter() - start_time, 3)
        average_confidence = self.calculate_average_confidence(text_boxes)

        return {
            "status": "OCR_SUCCESS",
            "cccdData": structured_data,
            "metadata": {
                "engine": "EasyOCR",
                "processingTime": processing_time,
                "averageConfidence": average_confidence,
                "validation": self.make_json_safe(ocr_result["validation"]),
                "inputImage": str(image_path),
                "cardImage": str(card_output_path),
                "enhancedImage": str(enhanced_output_path),
                "debugDir": str(debug_dir),
                "resizeRatio": self.make_json_safe(
                    detection_result["resizeRatio"]
                ),
            },
            "textBoxes": text_boxes,
        }

    def calculate_average_confidence(
        self,
        text_boxes: list[dict[str, Any]],
    ) -> float:
        if not text_boxes:
            return 0.0

        scores = [
            float(item.get("confidence", 0))
            for item in text_boxes
        ]

        return round(sum(scores) / len(scores), 4)

    def make_json_safe(self, value: Any) -> Any:
        """
        Chuyển dữ liệu numpy sang kiểu JSON có thể serialize được.
        """

        if isinstance(value, dict):
            return {
                key: self.make_json_safe(item)
                for key, item in value.items()
            }

        if isinstance(value, list):
            return [
                self.make_json_safe(item)
                for item in value
            ]

        if isinstance(value, tuple):
            return [
                self.make_json_safe(item)
                for item in value
            ]

        if hasattr(value, "item"):
            return value.item()

        return value


ocr_pipeline_service = OcrPipelineService()