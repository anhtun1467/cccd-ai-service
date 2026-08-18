from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.modules.card_detection.contour_detector import ContourDetector
from app.modules.card_detection.detector import CardDetector


def _build_far_shot() -> tuple[np.ndarray, np.ndarray]:
    """Tạo ảnh riêng tư giả lập bố cục thẻ nằm trên tay và cạnh màn hình."""
    height, width = 700, 528
    canvas = np.full((height, width, 3), (35, 45, 40), dtype=np.uint8)
    cv2.rectangle(canvas, (292, 0), (527, 180), (225, 225, 225), -1)
    cv2.ellipse(
        canvas,
        (260, 450),
        (380, 235),
        -8,
        0,
        360,
        (105, 155, 205),
        -1,
    )

    card = np.full((293, 465, 3), (190, 215, 190), dtype=np.uint8)
    cv2.rectangle(card, (1, 1), (463, 291), (55, 110, 95), 3)
    cv2.circle(card, (65, 55), 38, (25, 35, 210), -1)
    cv2.putText(
        card,
        "CAN CUOC CONG DAN",
        (125, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (35, 35, 210),
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(card, (20, 105), (125, 270), (110, 120, 130), -1)
    cv2.circle(card, (72, 153), 28, (145, 175, 205), -1)
    for y_value, line_length in (
        (112, 250),
        (142, 285),
        (172, 220),
        (202, 300),
        (232, 275),
        (262, 305),
    ):
        cv2.line(
            card,
            (145, y_value),
            (min(450, 145 + line_length), y_value),
            (45, 55, 55),
            2,
            cv2.LINE_AA,
        )
    for row in range(8):
        for column in range(8):
            if (column * 3 + row * 5) % 4 < 2:
                cv2.rectangle(
                    card,
                    (380 + column * 7, 12 + row * 7),
                    (386 + column * 7, 18 + row * 7),
                    (15, 15, 15),
                    -1,
                )

    source_corners = np.float32(
        [[0, 0], [464, 0], [464, 292], [0, 292]]
    )
    expected_corners = np.float32(
        [[25, 192], [489, 180], [494, 473], [35, 485]]
    )
    transform = cv2.getPerspectiveTransform(source_corners, expected_corners)
    warped_card = cv2.warpPerspective(card, transform, (width, height))
    card_mask = cv2.warpPerspective(
        np.full(card.shape[:2], 255, dtype=np.uint8),
        transform,
        (width, height),
    )
    canvas[card_mask > 0] = warped_card[card_mask > 0]
    return canvas, expected_corners


def test_far_shot_uses_four_edges_when_brightness_mask_merges_background(
    tmp_path: Path,
) -> None:
    image, expected_corners = _build_far_shot()
    image_path = tmp_path / "synthetic_far_shot.jpg"
    assert cv2.imwrite(str(image_path), image)

    result = CardDetector().detect_from_path(
        str(image_path),
        str(tmp_path / "debug"),
    )

    geometry = result["geometry"]
    actual = ContourDetector._order_quadrilateral(
        np.asarray(geometry["sourceCorners"], dtype=np.float32)
    )
    expected = ContourDetector._order_quadrilateral(expected_corners)
    mean_corner_error = float(
        np.mean(np.linalg.norm(actual - expected, axis=1))
    )

    assert geometry["detectionMethod"] == "hough_quadrilateral"
    assert mean_corner_error < 9.0
    assert geometry["sourceCoverageRatio"] > 0.34
    assert geometry["sourceCoverageRatio"] < 0.40
    assert abs(
        result["cardImage"].shape[1]
        / result["cardImage"].shape[0]
        - result["geometry"]["targetAspectRatio"]
    ) < 0.02


def test_far_shot_crop_keeps_original_resolution(tmp_path: Path) -> None:
    image, expected_corners = _build_far_shot()
    high_resolution = cv2.resize(
        image,
        None,
        fx=2.0,
        fy=2.0,
        interpolation=cv2.INTER_CUBIC,
    )
    image_path = tmp_path / "synthetic_far_shot_high_resolution.jpg"
    assert cv2.imwrite(str(image_path), high_resolution)

    result = CardDetector().detect_from_path(str(image_path))
    geometry = result["geometry"]
    actual = ContourDetector._order_quadrilateral(
        np.asarray(geometry["sourceCorners"], dtype=np.float32)
    )
    expected = ContourDetector._order_quadrilateral(
        expected_corners * 2.0
    )

    assert geometry["geometrySource"] == "original_image"
    assert float(np.mean(np.linalg.norm(actual - expected, axis=1))) < 18.0
    assert result["cardImage"].shape[1] > 850


def test_complete_card_skips_hough_even_with_strong_internal_rectangles(
    monkeypatch,
) -> None:
    image = np.full((700, 900, 3), 25, dtype=np.uint8)
    cv2.rectangle(image, (100, 130), (800, 570), (205, 220, 205), -1)
    cv2.rectangle(image, (100, 130), (800, 570), (45, 80, 70), 4)
    cv2.rectangle(image, (135, 255), (300, 525), (70, 70, 70), 5)
    cv2.rectangle(image, (660, 155), (765, 260), (10, 10, 10), 8)
    detector = ContourDetector()

    def fail_if_hough_runs(*args, **kwargs):
        raise AssertionError("Hough không được chạy khi contour đã trọn thẻ")

    monkeypatch.setattr(
        detector,
        "find_hough_card_quadrilaterals",
        fail_if_hough_runs,
    )
    contour, _, _, metadata = (
        detector.find_card_contour_candidates_from_image(image)
    )

    assert contour is not None
    assert metadata["wholeCardReliable"] is True
    assert metadata["houghFallbackEvaluated"] is False
    assert metadata["houghSkippedReason"] == "PRIMARY_CONTOUR_COMPLETE"
    assert metadata["alternateCandidates"] == []


def test_hough_refinement_uses_opposite_normals_for_full_card_edges() -> None:
    image = np.full((700, 525, 3), 25, dtype=np.uint8)
    expected = np.int32([
        [53, 321],
        [493, 317],
        [506, 603],
        [41, 607],
    ])
    cv2.fillConvexPoly(image, expected, (195, 215, 195))
    # Nền sáng nối vào đáy làm contour Otsu kéo xuống hết khung hình.
    cv2.rectangle(image, (0, 605), (524, 699), (195, 215, 195), -1)
    cv2.polylines(image, [expected], True, (40, 80, 70), 3)
    cv2.rectangle(image, (70, 370), (165, 555), (85, 100, 100), -1)
    for y_value in range(355, 580, 32):
        cv2.line(image, (185, y_value), (460, y_value), (40, 50, 50), 2)

    primary, _, _, metadata = (
        ContourDetector().find_card_contour_candidates_from_image(image)
    )

    assert metadata["detectionMethod"] == "hough_quadrilateral"
    replacement = metadata["contourReplacement"]
    assert replacement["reason"] == "SUSPICIOUS_BRIGHTNESS_CONTOUR"
    assert replacement["relativeArea"] > 0.65
    assert replacement["overlapWithContour"] > 0.95
    actual = ContourDetector._order_quadrilateral(primary)
    target = ContourDetector._order_quadrilateral(expected)
    assert float(np.mean(np.linalg.norm(actual - target, axis=1))) < 8.0


def test_large_card_touching_frame_is_not_replaced_by_internal_strip() -> None:
    """Hồi quy ảnh thật: contour gần trọn thẻ bị cụt góc có aspect 1.21."""
    image = np.full((700, 525, 3), 24, dtype=np.uint8)
    expected = np.int32([
        [0, 0],
        [500, 38],
        [384, 570],
        [38, 613],
    ])
    cv2.fillConvexPoly(image, expected, (205, 220, 205))
    cv2.polylines(image, [expected], True, (45, 80, 70), 3)

    # Chi tiết phân bố khắp thẻ, gồm một dải chữ có cạnh rất mạnh ở dưới.
    for y_value in range(70, 535, 55):
        left = 35 + int(y_value * 0.05)
        cv2.line(
            image,
            (left, y_value),
            (min(455, left + 330), y_value + 8),
            (35, 45, 45),
            3,
            cv2.LINE_AA,
        )
    cv2.rectangle(image, (52, 365), (393, 565), (35, 45, 45), 4)
    cv2.rectangle(image, (350, 55), (450, 155), (20, 20, 20), 7)

    detector = ContourDetector()
    primary, _, contours, metadata = (
        detector.find_card_contour_candidates_from_image(image)
    )

    assert contours
    assert detector.is_valid_card_contour(contours[0], image.shape) is False
    assert metadata["detectionMethod"] == "large_foreground_contour"
    assert metadata["wholeCardReliable"] is True
    assert metadata["houghFallbackEvaluated"] is False
    assert metadata["houghSkippedReason"] == (
        "LARGE_FOREGROUND_CONTOUR_COMPLETE"
    )
    assert metadata["primaryAreaRatio"] > 0.55
    actual = ContourDetector._order_quadrilateral(primary)
    target = ContourDetector._order_quadrilateral(expected)
    assert float(np.mean(np.linalg.norm(actual - target, axis=1))) < 14.0
