from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.modules.qr.qr_debug import (
    save_parser_qr_overlay,
    save_qr_debug_images,
)


def test_qr_debug_writes_detection_crop_and_parser_overlay(
    tmp_path: Path,
) -> None:
    card = np.full((630, 1000, 3), 235, dtype=np.uint8)
    qr_result = {
        "decoded": True,
        "regionDetected": True,
        "polygon": [[780, 45], [940, 45], [940, 205], [780, 205]],
        "boundingBox": {
            "x": 780,
            "y": 45,
            "width": 160,
            "height": 160,
        },
        "searchRegion": {
            "x": 600,
            "y": 0,
            "width": 400,
            "height": 300,
        },
        "debugCrop": card[35:215, 770:950].copy(),
    }

    debug_paths = save_qr_debug_images(card, qr_result, tmp_path)
    fields_path = tmp_path / "fields_debug.jpg"
    assert cv2.imwrite(str(fields_path), card)
    parser_overlay = save_parser_qr_overlay(
        fields_debug_path=fields_path,
        qr_result=qr_result,
        card_size=(1000, 630),
        output_dir=tmp_path,
    )

    assert Path(debug_paths["detectionImage"]).is_file()
    assert Path(debug_paths["cropImage"]).is_file()
    assert parser_overlay is not None
    assert Path(parser_overlay).is_file()


def test_qr_debug_draws_search_area_when_region_is_missing(
    tmp_path: Path,
) -> None:
    card = np.full((630, 1000, 3), 220, dtype=np.uint8)
    result = {
        "decoded": False,
        "regionDetected": False,
        "polygon": [],
        "searchRegion": {
            "x": 600,
            "y": 0,
            "width": 400,
            "height": 300,
        },
        "debugCrop": None,
    }

    paths = save_qr_debug_images(card, result, tmp_path)

    assert Path(paths["detectionImage"]).is_file()
    assert "cropImage" not in paths

