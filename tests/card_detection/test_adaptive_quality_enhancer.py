from __future__ import annotations

import cv2
import numpy as np

from app.modules.card_detection.adaptive_quality_enhancer import (
    AdaptiveQualityEnhancer,
)


def _text_image() -> np.ndarray:
    image = np.full((420, 760, 3), 224, dtype=np.uint8)
    for row, text in enumerate(
        (
            "CAN CUOC CONG DAN",
            "012345678901",
            "NGUYEN THI HUONG",
            "17/10/1991",
        )
    ):
        cv2.putText(
            image,
            text,
            (35, 90 + row * 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.25,
            (22, 22, 22),
            3,
            cv2.LINE_AA,
        )
    return image


def _laplacian_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def test_low_light_variant_is_brighter_without_changing_geometry() -> None:
    source = _text_image()
    dim = np.clip(
        source.astype(np.float32) * 0.34 + 2.0,
        0,
        255,
    ).astype(np.uint8)
    enhancer = AdaptiveQualityEnhancer()

    profile, variants = enhancer.build_ocr_variants(dim)
    low_light = next(
        item["image"]
        for item in variants
        if item["name"] == "low_light"
    )

    assert profile["isLowLight"] is True
    assert low_light.shape == dim.shape
    assert low_light.dtype == np.uint8
    assert float(np.mean(low_light)) > float(np.mean(dim)) + 35.0
    assert float(np.percentile(low_light, 99)) < 252.0


def test_mild_deblur_strengthens_text_edges_but_keeps_size() -> None:
    blurred = cv2.GaussianBlur(
        _text_image(),
        (7, 7),
        sigmaX=1.65,
        sigmaY=1.65,
    )
    enhancer = AdaptiveQualityEnhancer()

    profile, variants = enhancer.build_ocr_variants(blurred)
    deblurred = next(
        item["image"]
        for item in variants
        if item["name"] == "mild_deblur"
    )

    assert profile["isSlightlyBlurred"] is True
    assert deblurred.shape == blurred.shape
    assert _laplacian_score(deblurred) > _laplacian_score(blurred) * 1.15


def test_clear_image_is_not_processed_unnecessarily() -> None:
    enhancer = AdaptiveQualityEnhancer()

    profile, variants = enhancer.build_ocr_variants(_text_image())

    assert profile["needsEnhancement"] is False
    assert variants == []


def test_dark_and_blurred_image_gets_bounded_candidates() -> None:
    source = np.clip(
        _text_image().astype(np.float32) * 0.30,
        0,
        255,
    ).astype(np.uint8)
    source = cv2.GaussianBlur(source, (5, 5), 1.2)
    enhancer = AdaptiveQualityEnhancer()

    profile, variants = enhancer.build_ocr_variants(source)

    assert profile["isLowLight"] is True
    assert profile["isSlightlyBlurred"] is True
    assert [item["name"] for item in variants] == [
        "low_light_deblur",
        "low_light",
        "mild_deblur",
    ]
    assert all(item["image"].shape == source.shape for item in variants)
