from __future__ import annotations

import cv2
import numpy as np


class ImageEnhancer:
    """
    Image Enhancement trước OCR.

    Bao gồm:

    - Brightness
    - Contrast
    - CLAHE
    - Sharpen
    - Denoise
    """

    def adjust_brightness_contrast(
        self,
        image: np.ndarray,
        alpha: float = 1.15,
        beta: int = 8,
    ) -> np.ndarray:

        return cv2.convertScaleAbs(
            image,
            alpha=alpha,
            beta=beta,
        )

    def apply_clahe(
        self,
        image: np.ndarray,
    ) -> np.ndarray:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

        enhanced = clahe.apply(gray)

        return cv2.cvtColor(
            enhanced,
            cv2.COLOR_GRAY2BGR,
        )

    def sharpen(
        self,
        image: np.ndarray,
    ) -> np.ndarray:

        kernel = np.array(
            [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0],
            ]
        )

        return cv2.filter2D(
            image,
            -1,
            kernel,
        )

    def denoise(
        self,
        image: np.ndarray,
    ) -> np.ndarray:

        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            5,
            5,
            7,
            21,
        )

    def enhance(
        self,
        image: np.ndarray,
    ) -> dict[str, np.ndarray]:

        brightness = self.adjust_brightness_contrast(image)

        clahe = self.apply_clahe(brightness)

        sharpen = self.sharpen(clahe)

        denoise = self.denoise(sharpen)

        return {
            "brightness": brightness,
            "clahe": clahe,
            "sharpen": sharpen,
            "denoise": denoise,
            "final": denoise,
        }