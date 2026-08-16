from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.modules.ocr.field_cropper import CCCDFieldCropper


def test_dim_fields_keep_the_legacy_ocr_order(tmp_path: Path) -> None:
    card = np.full((630, 1000, 3), 58, dtype=np.uint8)
    cv2.putText(
        card,
        "NGUYEN THI HUONG",
        (290, 385),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (18, 18, 18),
        2,
        cv2.LINE_AA,
    )
    cropper = CCCDFieldCropper()

    results = cropper.crop_fields(
        image=card,
        output_dir=str(tmp_path / "fields"),
    )
    variants = results["fullName"]["variantImagePaths"]
    variant_names = [item["variant"] for item in variants]

    assert variant_names == [
        "value_processed",
        "value_raw",
        "value_detail",
        "wide_processed",
        "wide_raw",
    ]
    assert not any(
        "low_light" in name or "deblur" in name
        for name in variant_names
    )


def test_portrait_stays_unprocessed_for_face_verification(
    tmp_path: Path,
) -> None:
    card = np.full((630, 1000, 3), 70, dtype=np.uint8)
    cropper = CCCDFieldCropper()

    results = cropper.crop_fields(
        image=card,
        output_dir=str(tmp_path / "fields"),
    )
    portrait = results["portrait"]

    assert all(
        "low_light" not in item["variant"]
        and "deblur" not in item["variant"]
        for item in portrait["variantImagePaths"]
    )
