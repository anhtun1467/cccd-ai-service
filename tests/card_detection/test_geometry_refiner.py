from __future__ import annotations

import cv2
import numpy as np

from app.modules.card_detection.geometry_refiner import GeometryRefiner


def build_text_line_card() -> np.ndarray:
    image = np.full((630, 1000, 3), 232, dtype=np.uint8)
    for index, y_position in enumerate(range(120, 510, 55)):
        cv2.line(
            image,
            (300, y_position),
            (790, y_position),
            (25, 25, 25),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"HO VA TEN NGUYEN VAN A {index}",
            (310, y_position - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (35, 35, 35),
            1,
            cv2.LINE_AA,
        )
    return image


def test_hough_estimates_and_corrects_positive_text_skew() -> None:
    refiner = GeometryRefiner()
    horizontal = build_text_line_card()

    # Dùng shear ngược để tạo ảnh có baseline dốc xuống về bên phải.
    skewed = refiner.correct_vertical_shear(horizontal, -5.0)
    estimate = refiner.estimate_text_skew(skewed)

    assert estimate["reliable"] is True
    assert 3.5 <= estimate["angleDegrees"] <= 6.5

    corrected = refiner.correct_vertical_shear(
        skewed,
        estimate["angleDegrees"],
    )
    residual = refiner.estimate_text_skew(corrected)

    assert abs(residual["angleDegrees"]) < 1.25


def test_axis_aligned_easyocr_boxes_do_not_cancel_hough_skew() -> None:
    refiner = GeometryRefiner()
    horizontal = build_text_line_card()
    skewed = refiner.correct_vertical_shear(horizontal, -4.0)
    axis_aligned_boxes = [
        {
            "text": "NGUYEN VAN A",
            "confidence": 0.9,
            "box": [
                [260.0, float(y_position)],
                [720.0, float(y_position)],
                [720.0, float(y_position + 30)],
                [260.0, float(y_position + 30)],
            ],
        }
        for y_position in (120, 200, 280, 360)
    ]

    estimate = refiner.estimate_text_skew(
        skewed,
        axis_aligned_boxes,
    )

    assert estimate["source"] == "hough"
    assert 3.0 <= estimate["angleDegrees"] <= 5.0


def test_ocr_boxes_are_used_when_hough_has_no_visual_lines() -> None:
    refiner = GeometryRefiner()
    blank = np.full((630, 1000, 3), 220, dtype=np.uint8)
    slope = np.tan(np.deg2rad(4.0))
    boxes = []
    for y_position in (100, 180, 260, 340):
        left = 250.0
        right = 650.0
        top_left_y = float(y_position)
        top_right_y = top_left_y + (right - left) * slope
        boxes.append(
            {
                "text": "NGUYEN VAN A",
                "confidence": 0.9,
                "box": [
                    [left, top_left_y],
                    [right, top_right_y],
                    [right, top_right_y + 30.0],
                    [left, top_left_y + 30.0],
                ],
            }
        )

    estimate = refiner.estimate_text_skew(blank, boxes)

    assert estimate["reliable"] is True
    assert estimate["source"] == "ocr_boxes"
    assert abs(estimate["angleDegrees"] - 4.0) < 0.4


def test_blank_card_does_not_trigger_skew_retry() -> None:
    refiner = GeometryRefiner()
    blank = np.full((630, 1000, 3), 220, dtype=np.uint8)

    estimate = refiner.estimate_text_skew(blank)

    assert estimate["reliable"] is False
    assert refiner.build_correction_angles(estimate) == []
