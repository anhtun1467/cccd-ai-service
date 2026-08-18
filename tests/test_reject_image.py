from __future__ import annotations

import cv2
import numpy as np

from app.utils.image_validator import (
    INPUT_HARD_DARK_THRESHOLD,
    check_image_quality,
)


def _checkerboard(
    height: int = 600,
    width: int = 900,
    block_size: int = 20,
) -> np.ndarray:
    rows, columns = np.indices((height, width))
    mask = ((rows // block_size) + (columns // block_size)) % 2
    gray = np.where(mask == 0, 20, 220).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_clear_image_is_accepted() -> None:
    result = check_image_quality(_checkerboard())

    assert result["is_valid"] is True
    assert result["error_code"] == "VALID"


def test_empty_and_unsupported_images_return_structured_errors() -> None:
    empty = check_image_quality(np.array([], dtype=np.uint8))
    unsupported = check_image_quality(np.zeros((100, 100, 2), dtype=np.uint8))

    assert empty["error_code"] == "INVALID_IMAGE"
    assert unsupported["error_code"] == "UNSUPPORTED_IMAGE_CHANNELS"


def test_crop_stage_can_use_a_stricter_blur_threshold() -> None:
    blurred = cv2.GaussianBlur(_checkerboard(), (31, 31), 9)
    baseline = check_image_quality(blurred, blur_threshold=0.0)

    input_stage = check_image_quality(
        blurred,
        blur_threshold=max(0.0, baseline["blur_score"] - 1.0),
    )
    crop_stage = check_image_quality(
        blurred,
        blur_threshold=baseline["blur_score"] + 1.0,
    )

    assert input_stage["is_valid"] is True
    assert crop_stage["is_valid"] is False
    assert crop_stage["error_code"] == "BLURRY_IMAGE"


def test_dim_but_visible_input_reaches_the_ocr_pipeline() -> None:
    dim_image = np.clip(
        _checkerboard().astype(np.float32) * 0.22,
        0,
        255,
    ).astype(np.uint8)

    result = check_image_quality(
        dim_image,
        blur_threshold=0.0,
        dark_threshold=INPUT_HARD_DARK_THRESHOLD,
    )

    assert INPUT_HARD_DARK_THRESHOLD < result["brightness_score"] < 60.0
    assert result["is_valid"] is True


def test_nearly_black_input_is_still_rejected() -> None:
    nearly_black = np.full((240, 360, 3), 5, dtype=np.uint8)

    result = check_image_quality(
        nearly_black,
        blur_threshold=0.0,
        dark_threshold=INPUT_HARD_DARK_THRESHOLD,
    )

    assert result["is_valid"] is False
    assert result["error_code"] == "DARK_IMAGE"
