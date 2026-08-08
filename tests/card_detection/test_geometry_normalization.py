from __future__ import annotations

import cv2
import numpy as np

from app.modules.card_detection.contour_detector import ContourDetector
from app.modules.card_detection.perspective_transformer import (
    PerspectiveTransformer,
)


def test_contour_keeps_real_trapezoid_corners() -> None:
    detector = ContourDetector(padding=0)
    contour = np.array(
        [
            [[80, 90]],
            [[540, 45]],
            [[590, 360]],
            [[35, 405]],
        ],
        dtype=np.int32,
    )

    quadrilateral = detector.contour_to_quadrilateral(
        contour,
        image_shape=(480, 640, 3),
    )

    assert quadrilateral.shape == (4, 1, 2)
    actual = {tuple(point) for point in quadrilateral.reshape(4, 2)}
    expected = {tuple(point) for point in contour.reshape(4, 2)}
    assert actual == expected


def test_perspective_transform_flattens_trapezoid_to_card_ratio() -> None:
    transformer = PerspectiveTransformer()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    points = np.array(
        [
            [[80, 90]],
            [[540, 45]],
            [[590, 360]],
            [[35, 405]],
        ],
        dtype=np.float32,
    )

    warped, metadata = transformer.transform_with_metadata(image, points)
    height, width = warped.shape[:2]

    assert width > height
    assert abs((width / height) - transformer.CARD_ASPECT_RATIO) < 0.02
    assert metadata["perspectiveApplied"] is True
    assert metadata["geometryRotationDegrees"] == 0


def test_portrait_card_is_rotated_to_landscape_after_warp() -> None:
    transformer = PerspectiveTransformer()
    image = np.zeros((720, 480, 3), dtype=np.uint8)
    points = np.array(
        [
            [[70, 40]],
            [[410, 65]],
            [[390, 660]],
            [[45, 635]],
        ],
        dtype=np.float32,
    )

    warped, metadata = transformer.transform_with_metadata(image, points)

    assert warped.shape[1] > warped.shape[0]
    assert metadata["geometryRotationDegrees"] == 90


def test_rotated_contour_ratio_is_orientation_independent() -> None:
    detector = ContourDetector()
    rect = ((240.0, 350.0), (560.0, 350.0), 82.0)
    contour = cv2.boxPoints(rect).reshape(4, 1, 2).astype(np.int32)

    assert detector.is_valid_card_contour(
        contour,
        image_shape=(700, 480, 3),
    )


def test_two_portrait_cards_with_shared_seam_are_detected() -> None:
    detector = ContourDetector()
    image = np.full((700, 960, 3), 205, dtype=np.uint8)

    # Hai thẻ dọc đặt sát nhau; morphology lớn có thể nối chúng thành
    # một contour, nên fallback dựa trên đường phân cách phải bắt được.
    cv2.line(image, (494, 0), (494, 699), (35, 35, 35), 7)
    regions = detector.find_tiled_card_regions(image)

    assert len(regions) == 2
    assert all(region.shape == (4, 1, 2) for region in regions)


def test_single_landscape_card_is_not_reported_as_two_cards() -> None:
    detector = ContourDetector()
    image = np.full((630, 1000, 3), 205, dtype=np.uint8)
    cv2.rectangle(image, (20, 180), (260, 520), (80, 80, 80), 3)

    assert detector.find_tiled_card_regions(image) == []
