from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _invalid_result(reason: str, error_code: str) -> dict[str, Any]:
    return {
        "is_valid": False,
        "is_blurry": False,
        "is_too_dark": False,
        "blur_score": 0.0,
        "brightness_score": 0.0,
        "reason": reason,
        "error_code": error_code,
        "suggestion": "Vui lòng gửi ảnh JPG, JPEG hoặc PNG hợp lệ.",
    }


def check_image_quality(
    img: np.ndarray,
    blur_threshold: float = 40.0,
    dark_threshold: float = 60.0,
) -> dict[str, Any]:
    """Chấm độ nét và độ sáng mà không làm API lỗi với ảnh hỏng.

    Ngưỡng mặc định chỉ chặn ảnh đầu vào *rất mờ*. Sau khi CardDetector
    cắt đúng một mặt thẻ, pipeline gọi lại hàm với ngưỡng 80 để đánh giá
    riêng vùng CCCD. Cách kiểm tra hai giai đoạn tránh trường hợp nền hoặc
    hai thẻ trong cùng ảnh làm điểm Laplacian toàn ảnh bị thấp giả tạo.
    """
    if not isinstance(img, np.ndarray) or img.size == 0:
        return _invalid_result("Ảnh rỗng hoặc không đọc được", "INVALID_IMAGE")

    if img.ndim == 2:
        gray_img = img
    elif img.ndim == 3 and img.shape[2] == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3 and img.shape[2] == 4:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        return _invalid_result(
            "Định dạng kênh màu của ảnh không được hỗ trợ",
            "UNSUPPORTED_IMAGE_CHANNELS",
        )

    blur_score = float(cv2.Laplacian(gray_img, cv2.CV_64F).var())
    brightness_score = float(np.mean(gray_img))
    is_blurry = blur_score < float(blur_threshold)
    is_too_dark = brightness_score < float(dark_threshold)

    if is_blurry and is_too_dark:
        reason = "Ảnh quá mờ và thiếu sáng"
        error_code = "BLURRY_AND_DARK"
    elif is_blurry:
        reason = "Ảnh quá mờ"
        error_code = "BLURRY_IMAGE"
    elif is_too_dark:
        reason = "Ảnh thiếu sáng"
        error_code = "DARK_IMAGE"
    else:
        reason = "Hợp lệ"
        error_code = "VALID"

    return {
        "is_valid": not (is_blurry or is_too_dark),
        "is_blurry": is_blurry,
        "is_too_dark": is_too_dark,
        "blur_score": round(blur_score, 2),
        "brightness_score": round(brightness_score, 2),
        "reason": reason,
        "error_code": error_code,
        "suggestion": (
            "Vui lòng chụp lại ảnh rõ nét, đủ sáng và giữ CCCD ổn định."
            if is_blurry or is_too_dark
            else None
        ),
    }
