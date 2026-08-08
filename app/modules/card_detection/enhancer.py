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
        alpha: float = 1.08,
        beta: int = 3,
    ) -> np.ndarray:
        # Không đẩy sáng mạnh ảnh CCCD vốn đã có nền vàng/trắng. Cháy nền
        # làm mất các dấu thanh nhỏ trước khi OCR kịp nhận dạng.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if float(np.mean(gray)) >= 175.0:
            alpha = min(alpha, 1.02)
            beta = min(beta, 1)
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
        # Kernel 5 điểm cũ tạo viền đen/trắng dày quanh chữ mờ. Unsharp
        # mask nhẹ giữ được nét dấu tiếng Việt và ít tạo ký tự giả hơn.
        blurred = cv2.GaussianBlur(
            image,
            (0, 0),
            sigmaX=0.85,
            sigmaY=0.85,
        )
        return cv2.addWeighted(
            image,
            1.35,
            blurred,
            -0.35,
            0,
        )

    def denoise(
        self,
        image: np.ndarray,
    ) -> np.ndarray:

        return cv2.bilateralFilter(
            image,
            d=5,
            sigmaColor=25,
            sigmaSpace=25,
        )

    def enhance(
        self,
        image: np.ndarray,
    ) -> dict[str, np.ndarray]:

        brightness = self.adjust_brightness_contrast(image)

        # Khử nhiễu trước khi tăng tương phản/nét để không khuếch đại hoa
        # văn bảo an thành các nét giống chữ.
        denoise = self.denoise(brightness)

        clahe = self.apply_clahe(denoise)

        sharpen = self.sharpen(clahe)

        return {
            "brightness": brightness,
            "clahe": clahe,
            "sharpen": sharpen,
            "denoise": denoise,
            "final": sharpen,
        }
