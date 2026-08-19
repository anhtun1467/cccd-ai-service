from __future__ import annotations

import cv2
import numpy as np

from app.modules.card_detection.fast_orientation import FastCardOrientation


def _synthetic_front() -> np.ndarray:
    image = np.full((630, 1000, 3), (205, 225, 215), dtype=np.uint8)
    # Quốc huy và tiêu đề đỏ nằm ở nửa trên của mặt trước.
    cv2.circle(image, (145, 155), 62, (25, 35, 220), -1)
    cv2.rectangle(image, (330, 215), (790, 255), (20, 30, 215), -1)
    # Chi tiết chữ tối trải đều để ảnh gần với thẻ thật hơn.
    for y_value in range(300, 590, 42):
        cv2.line(image, (300, y_value), (920, y_value), (45, 55, 55), 3)
    return image


def test_red_layout_locks_upright_before_ocr() -> None:
    result = FastCardOrientation().analyze(_synthetic_front())

    assert result["reliable"] is True
    assert result["rotationDegrees"] == 0
    assert result["source"] == "FRONT_RED_LAYOUT"


def test_red_layout_detects_upside_down_before_ocr() -> None:
    upside_down = cv2.rotate(_synthetic_front(), cv2.ROTATE_180)

    result = FastCardOrientation().analyze(upside_down)

    assert result["reliable"] is True
    assert result["rotationDegrees"] == 180
    assert result["source"] == "FRONT_RED_LAYOUT"


def test_undecoded_qr_region_can_lock_bottom_left_orientation() -> None:
    image = np.full((630, 1000, 3), 190, dtype=np.uint8)
    qr_result = {
        "decoded": False,
        "regionDetected": True,
        "boundingBox": {"x": 85, "y": 465, "width": 145, "height": 145},
    }

    result = FastCardOrientation().analyze(image, qr_result=qr_result)

    assert result["reliable"] is True
    assert result["rotationDegrees"] == 180
    assert result["source"] == "QR_REGION_POSITION"


def test_ambiguous_gray_card_keeps_ocr_orientation_fallback() -> None:
    image = np.full((630, 1000, 3), 180, dtype=np.uint8)

    result = FastCardOrientation().analyze(image)

    assert result["reliable"] is False
    assert result["rotationDegrees"] == 0
