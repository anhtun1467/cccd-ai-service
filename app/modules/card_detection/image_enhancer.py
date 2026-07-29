from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class CCCDImageEnhancer:
    """Tăng cường ảnh CCCD trước khi OCR."""

    def __init__(
        self,
        target_width: int = 1600,
        clahe_clip_limit: float = 2.0,
        sharpen_amount: float = 1.15,
    ) -> None:
        self.target_width = target_width
        self.clahe_clip_limit = clahe_clip_limit
        self.sharpen_amount = sharpen_amount

    def enhance(self, image: np.ndarray) -> np.ndarray:
        self._validate_image(image)

        resized = self._resize_if_needed(image)
        denoised = cv2.bilateralFilter(
            resized,
            d=5,
            sigmaColor=30,
            sigmaSpace=30,
        )

        lab = cv2.cvtColor(
            denoised,
            cv2.COLOR_BGR2LAB,
        )
        lightness, channel_a, channel_b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=(8, 8),
        )
        lightness = clahe.apply(lightness)

        contrast = cv2.cvtColor(
            cv2.merge((lightness, channel_a, channel_b)),
            cv2.COLOR_LAB2BGR,
        )

        blurred = cv2.GaussianBlur(
            contrast,
            (0, 0),
            sigmaX=1.1,
            sigmaY=1.1,
        )

        sharpened = cv2.addWeighted(
            contrast,
            1.0 + self.sharpen_amount,
            blurred,
            -self.sharpen_amount,
            0,
        )

        return np.clip(
            sharpened,
            0,
            255,
        ).astype(np.uint8)

    def estimate_blur(
        self,
        image: np.ndarray,
    ) -> dict[str, Any]:
        self._validate_image(image)

        gray = self._to_gray(image)
        score = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F,
            ).var()
        )

        if score < 35:
            level = "VERY_BLURRY"
        elif score < 80:
            level = "BLURRY"
        elif score < 150:
            level = "ACCEPTABLE"
        else:
            level = "SHARP"

        return {
            "blurScore": round(score, 2),
            "blurLevel": level,
        }

    def _resize_if_needed(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        height, width = image.shape[:2]

        if width >= self.target_width:
            return image.copy()

        ratio = self.target_width / float(width)
        target_height = max(
            1,
            int(round(height * ratio)),
        )

        return cv2.resize(
            image,
            (self.target_width, target_height),
            interpolation=cv2.INTER_CUBIC,
        )

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image.copy()

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if image is None:
            raise ValueError("Ảnh đầu vào không được là None")

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "Ảnh đầu vào phải là numpy.ndarray"
            )

        if image.size == 0:
            raise ValueError(
                "Ảnh đầu vào không có dữ liệu"
            )


cccd_image_enhancer = CCCDImageEnhancer()