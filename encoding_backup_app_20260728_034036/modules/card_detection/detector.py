from pathlib import Path

import cv2
import numpy as np

from app.core.exceptions import BadRequestException
from app.modules.card_detection.contour_detector import ContourDetector
from app.modules.card_detection.enhancer import ImageEnhancer
from app.modules.card_detection.image_loader import ImageLoader
from app.modules.card_detection.perspective_transformer import PerspectiveTransformer
from app.modules.card_detection.preprocessor import ImagePreprocessor


class CardDetector:
    """
    Pipeline phát hiện và chuẩn hóa ảnh CCCD.
    """

    def __init__(self) -> None:
        self.loader = ImageLoader()
        self.preprocessor = ImagePreprocessor()
        self.contour_detector = ContourDetector()
        self.transformer = PerspectiveTransformer()
        self.enhancer = ImageEnhancer()

    def detect_from_path(
        self,
        image_path: str,
        output_dir: str | None = None,
    ) -> dict:
        image = self.loader.load(image_path)

        resized, ratio = self.preprocessor.resize(image)

        card_contour, mask, contours = self.contour_detector.find_card_contour_from_image(
            resized
        )

        if card_contour is None:
            raise BadRequestException("Không phát hiện được vùng CCCD")

        warped = self.transformer.transform(resized, card_contour)
        enhanced_images = self.enhancer.enhance(warped)

        result = {
            "success": True,
            "message": "Phát hiện CCCD thành công",
            "resizeRatio": ratio,
            "cardImage": warped,
            "enhancedImage": enhanced_images["final"],
            "debug": {
                "resized": resized,
                "mask": mask,
                "warped": warped,
                "enhanced": enhanced_images,
            },
        }

        if output_dir:
            self.save_debug_images(result, output_dir)

        return result

    def save_debug_images(self, result: dict, output_dir: str) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_path / "detector_01_resized.jpg"), result["debug"]["resized"])
        cv2.imwrite(str(output_path / "detector_02_mask.jpg"), result["debug"]["mask"])
        cv2.imwrite(str(output_path / "detector_03_warped.jpg"), result["debug"]["warped"])
        cv2.imwrite(str(output_path / "detector_04_enhanced.jpg"), result["debug"]["enhanced"]["final"])
