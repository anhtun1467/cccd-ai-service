from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def check_image_quality(
    image: np.ndarray,
    blur_threshold: float = 100.0,
    dark_threshold: float = 60.0,
) -> dict[str, Any]:
    """
    Kiểm tra chất lượng ảnh trước khi đưa vào OCR.

    Ảnh bị từ chối khi:
    - Độ nét thấp hơn blur_threshold.
    - Độ sáng thấp hơn dark_threshold.
    """

    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return {
            "is_valid": False,
            "blur_score": 0.0,
            "brightness_score": 0.0,
            "reason": "Không thể đọc được ảnh hoặc ảnh không có dữ liệu",
        }

    try:
        # Hỗ trợ ảnh xám, BGR và BGRA
        if image.ndim == 2:
            gray_image = image
        elif image.ndim == 3 and image.shape[2] == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 3 and image.shape[2] == 4:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            return {
                "is_valid": False,
                "blur_score": 0.0,
                "brightness_score": 0.0,
                "reason": "Định dạng hoặc số kênh màu của ảnh không được hỗ trợ",
            }

        # Phương sai Laplacian càng thấp thì ảnh càng mờ
        blur_score = float(
            cv2.Laplacian(gray_image, cv2.CV_64F).var()
        )

        # Độ sáng trung bình trong khoảng 0–255
        brightness_score = float(np.mean(gray_image))

        is_blurry = blur_score < blur_threshold
        is_too_dark = brightness_score < dark_threshold

        reasons: list[str] = []

        if is_blurry:
            reasons.append("Ảnh quá mờ")

        if is_too_dark:
            reasons.append("Ảnh thiếu sáng")

        is_valid = not reasons
        reason = "Hợp lệ" if is_valid else " và ".join(reasons)

        return {
            "is_valid": is_valid,
            "blur_score": round(blur_score, 2),
            "brightness_score": round(brightness_score, 2),
            "reason": reason,
        }

    except cv2.error:
        return {
            "is_valid": False,
            "blur_score": 0.0,
            "brightness_score": 0.0,
            "reason": "OpenCV không thể kiểm tra chất lượng ảnh",
        }