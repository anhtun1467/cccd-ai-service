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

        multiple_cards, multiple_mask = (
            self.contour_detector.find_multiple_card_contours_from_image(
                resized
            )
        )

        if len(multiple_cards) >= 2:
            if output_dir:
                self.save_multiple_card_debug(
                    resized,
                    multiple_mask,
                    multiple_cards,
                    output_dir,
                )
            raise BadRequestException(
                "Phát hiện nhiều CCCD trong cùng một ảnh",
                data={
                    "errorCode": "MULTIPLE_CARDS",
                    "reason": "Phát hiện nhiều CCCD trong cùng một ảnh",
                    "cardCount": len(multiple_cards),
                    "suggestion": (
                        "Vui lòng chỉ chụp một CCCD trong mỗi ảnh."
                    ),
                },
            )

        card_contour, mask, contours = self.contour_detector.find_card_contour_from_image(
            resized
        )

        if card_contour is None:
            raise BadRequestException("Không phát hiện được vùng CCCD")

        warped, geometry = self.transformer.transform_with_metadata(
            resized,
            card_contour,
        )
        enhanced_images = self.enhancer.enhance(warped)

        result = {
            "success": True,
            "message": "Phát hiện CCCD thành công",
            "resizeRatio": ratio,
            "geometry": geometry,
            "cardCount": 1,
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

    @staticmethod
    def save_multiple_card_debug(
        resized: np.ndarray,
        mask: np.ndarray,
        card_contours: list[np.ndarray],
        output_dir: str,
    ) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        overlay = resized.copy()
        debug_contours = [
            np.rint(contour).astype(np.int32)
            for contour in card_contours
        ]
        cv2.drawContours(overlay, debug_contours, -1, (0, 0, 255), 5)
        for index, contour in enumerate(debug_contours, start=1):
            x, y, _, _ = cv2.boundingRect(contour)
            cv2.putText(
                overlay,
                f"CCCD {index}",
                (x, max(35, y + 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imwrite(
            str(output_path / "detector_00_multiple_mask.jpg"),
            mask,
        )
        cv2.imwrite(
            str(output_path / "detector_00_multiple_cards.jpg"),
            overlay,
        )

    def save_debug_images(self, result: dict, output_dir: str) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_path / "detector_01_resized.jpg"), result["debug"]["resized"])
        cv2.imwrite(str(output_path / "detector_02_mask.jpg"), result["debug"]["mask"])
        cv2.imwrite(str(output_path / "detector_03_warped.jpg"), result["debug"]["warped"])
        cv2.imwrite(str(output_path / "detector_04_enhanced.jpg"), result["debug"]["enhanced"]["final"])
