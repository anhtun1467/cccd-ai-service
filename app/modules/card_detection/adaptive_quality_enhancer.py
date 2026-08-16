from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


class AdaptiveQualityEnhancer:
    """Tạo các biến thể OCR riêng cho ảnh thiếu sáng hoặc mờ nhẹ.

    Ảnh trả về luôn giữ nguyên kích thước ảnh đầu vào. Điều này rất quan
    trọng vì tọa độ text box của OCR toàn thẻ còn được dùng bởi layout
    parser và bộ hợp nhất kết quả.
    """

    DEFAULT_LOW_LIGHT_THRESHOLD = 128.0
    DEFAULT_BLUR_THRESHOLD = 110.0
    VERY_LOW_LIGHT_THRESHOLD = 55.0
    SEVERE_BLUR_THRESHOLD = 30.0

    def analyze(
        self,
        image: np.ndarray,
        low_light_threshold: float | None = None,
        blur_threshold: float | None = None,
    ) -> dict[str, Any]:
        self._validate_image(image)
        gray = self._to_gray(image)
        brightness = float(np.mean(gray))
        median = float(np.median(gray))
        percentile_10 = float(np.percentile(gray, 10))
        percentile_90 = float(np.percentile(gray, 90))
        blur_score = float(
            cv2.Laplacian(gray, cv2.CV_64F).var()
        )

        dark_limit = float(
            self.DEFAULT_LOW_LIGHT_THRESHOLD
            if low_light_threshold is None
            else low_light_threshold
        )
        blur_limit = float(
            self.DEFAULT_BLUR_THRESHOLD
            if blur_threshold is None
            else blur_threshold
        )
        # p90 thấp phát hiện được ảnh thiếu sáng cục bộ dù nền hoặc vùng
        # phản quang làm mean tăng giả. Điều kiện này chỉ tạo ứng viên OCR,
        # không tự động loại ảnh hay ghi đè ảnh gốc.
        is_low_light = bool(
            brightness < dark_limit
            or percentile_90 < max(165.0, dark_limit + 42.0)
        )
        is_slightly_blurred = bool(blur_score < blur_limit)

        return {
            "brightnessScore": round(brightness, 2),
            "medianBrightness": round(median, 2),
            "percentile10": round(percentile_10, 2),
            "percentile90": round(percentile_90, 2),
            "dynamicRange": round(percentile_90 - percentile_10, 2),
            "contrastStdDev": round(float(np.std(gray)), 2),
            "blurScore": round(blur_score, 2),
            "lowLightThreshold": round(dark_limit, 2),
            "blurThreshold": round(blur_limit, 2),
            "isLowLight": is_low_light,
            "isVeryLowLight": brightness < self.VERY_LOW_LIGHT_THRESHOLD,
            "isSlightlyBlurred": is_slightly_blurred,
            "isSeverelyBlurred": blur_score < self.SEVERE_BLUR_THRESHOLD,
            "needsEnhancement": bool(
                is_low_light or is_slightly_blurred
            ),
        }

    def enhance_low_light(
        self,
        image: np.ndarray,
        profile: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Nâng vùng tối trên kênh sáng, giữ màu và tránh cháy nền thẻ."""
        self._validate_image(image)
        quality = profile or self.analyze(image)
        bgr, restore = self._as_bgr(image)

        # Khử nhiễu rất nhẹ trước khi nâng sáng để nhiễu ISO không biến
        # thành những chấm giống dấu tiếng Việt.
        if bool(quality.get("isVeryLowLight")):
            working = cv2.bilateralFilter(
                bgr,
                d=5,
                sigmaColor=24,
                sigmaSpace=24,
            )
        else:
            working = cv2.bilateralFilter(
                bgr,
                d=3,
                sigmaColor=16,
                sigmaSpace=16,
            )

        lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        median_lightness = max(float(np.median(lightness)), 1.0)
        target = (
            154.0
            if float(quality.get("brightnessScore", 255.0)) < 82.0
            else 146.0
        )
        denominator = math.log(median_lightness / 255.0)
        gamma = (
            0.94
            if abs(denominator) < 1e-6
            else math.log(target / 255.0) / denominator
        )
        gamma = float(np.clip(gamma, 0.50, 0.94))
        lookup = np.array(
            [
                np.clip(((value / 255.0) ** gamma) * 255.0, 0, 255)
                for value in range(256)
            ],
            dtype=np.uint8,
        )
        gamma_lightness = cv2.LUT(lightness, lookup)

        clahe = cv2.createCLAHE(
            clipLimit=(
                1.9 if bool(quality.get("isVeryLowLight")) else 1.55
            ),
            tileGridSize=(8, 8),
        )
        local_contrast = clahe.apply(gamma_lightness)
        corrected_lightness = cv2.addWeighted(
            gamma_lightness,
            0.72,
            local_contrast,
            0.28,
            0,
        )
        corrected = cv2.cvtColor(
            cv2.merge(
                (corrected_lightness, channel_a, channel_b)
            ),
            cv2.COLOR_LAB2BGR,
        )
        return restore(corrected)

    def enhance_mild_blur(
        self,
        image: np.ndarray,
        profile: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Tăng nét có ngưỡng, dành cho rung/mất nét nhẹ.

        Đây không phải deconvolution mạnh. Phần chênh lệch rất nhỏ bị bỏ
        qua để không khuếch đại nền bảo an và không tạo viền kép quanh chữ.
        """
        self._validate_image(image)
        quality = profile or self.analyze(image)
        bgr, restore = self._as_bgr(image)

        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        softened = cv2.GaussianBlur(
            lightness,
            (0, 0),
            sigmaX=0.85,
            sigmaY=0.85,
        )
        blur_score = float(quality.get("blurScore", 110.0))
        if blur_score < 35.0:
            amount = 0.62
        elif blur_score < 70.0:
            amount = 0.50
        else:
            amount = 0.36

        original_float = lightness.astype(np.float32)
        detail = original_float - softened.astype(np.float32)
        detail[np.abs(detail) < 1.75] = 0.0
        sharpened_lightness = np.clip(
            original_float + amount * detail,
            0,
            255,
        ).astype(np.uint8)
        sharpened = cv2.cvtColor(
            cv2.merge(
                (sharpened_lightness, channel_a, channel_b)
            ),
            cv2.COLOR_LAB2BGR,
        )
        return restore(sharpened)

    def enhance_balanced(self, image: np.ndarray) -> np.ndarray:
        """Ứng viên nhẹ khi OCR yếu nhưng chỉ số ảnh chưa vượt ngưỡng."""
        self._validate_image(image)
        bgr, restore = self._as_bgr(image)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        local = cv2.createCLAHE(
            clipLimit=1.25,
            tileGridSize=(8, 8),
        ).apply(lightness)
        blended = cv2.addWeighted(lightness, 0.68, local, 0.32, 0)
        balanced = cv2.cvtColor(
            cv2.merge((blended, channel_a, channel_b)),
            cv2.COLOR_LAB2BGR,
        )
        balanced_profile = self.analyze(balanced)
        return self.enhance_mild_blur(
            restore(balanced),
            profile=balanced_profile,
        )

    def build_ocr_variants(
        self,
        image: np.ndarray,
        force_balanced: bool = False,
        low_light_threshold: float | None = None,
        blur_threshold: float | None = None,
        maximum_variants: int = 3,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Trả về profile và tối đa ba ảnh ứng viên cùng kích thước."""
        profile = self.analyze(
            image,
            low_light_threshold=low_light_threshold,
            blur_threshold=blur_threshold,
        )
        variants: list[dict[str, Any]] = []

        is_dark = bool(profile["isLowLight"])
        is_blurred = bool(profile["isSlightlyBlurred"])
        low_light_image: np.ndarray | None = None

        if is_dark:
            low_light_image = self.enhance_low_light(
                image,
                profile=profile,
            )
        if is_dark and is_blurred and low_light_image is not None:
            variants.append({
                "name": "low_light_deblur",
                "image": self.enhance_mild_blur(
                    low_light_image,
                    profile=profile,
                ),
            })
        if is_dark and low_light_image is not None:
            variants.append({
                "name": "low_light",
                "image": low_light_image,
            })
        if is_blurred:
            variants.append({
                "name": "mild_deblur",
                "image": self.enhance_mild_blur(
                    image,
                    profile=profile,
                ),
            })
        if not variants and force_balanced:
            variants.append({
                "name": "balanced",
                "image": self.enhance_balanced(image),
            })

        limit = max(0, int(maximum_variants))
        return profile, variants[:limit]

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _as_bgr(
        image: np.ndarray,
    ) -> tuple[np.ndarray, Any]:
        if image.ndim == 2:
            return (
                cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
                lambda value: cv2.cvtColor(value, cv2.COLOR_BGR2GRAY),
            )
        if image.shape[2] == 4:
            alpha = image[:, :, 3].copy()

            def restore_alpha(value: np.ndarray) -> np.ndarray:
                return np.dstack((value, alpha))

            return image[:, :, :3].copy(), restore_alpha
        return image.copy(), lambda value: value

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if image is None or not isinstance(image, np.ndarray):
            raise TypeError("Ảnh phải là numpy.ndarray")
        if image.size == 0:
            raise ValueError("Ảnh không có dữ liệu")
        if image.ndim == 2:
            return
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError("Chỉ hỗ trợ ảnh xám, BGR hoặc BGRA")


adaptive_quality_enhancer = AdaptiveQualityEnhancer()
