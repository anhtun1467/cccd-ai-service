from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.modules.ocr.field_cropper import CCCDFieldCropper
from app.modules.ocr.field_ocr_service import FieldOCRService
from app.modules.ocr.result_fuser import estimate_field_crop_layout


def _box(text: str, x1: int, y1: int, x2: int, y2: int) -> dict:
    return {
        "text": text,
        "confidence": 0.90,
        "box": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
    }


def test_label_anchors_create_non_overlapping_field_rows() -> None:
    text_boxes = [
        _box("Số / No.: 042096015766", 300, 255, 770, 300),
        _box("Họ và tên / Full name:", 300, 315, 600, 340),
        _box("Ngày sinh / Date of birth:", 300, 385, 700, 410),
        _box(
            "Giới tính / Sex: Nam  Quốc tịch / Nationality: Việt Nam",
            300,
            420,
            960,
            447,
        ),
        _box("Quê quán / Place of origin:", 300, 460, 760, 482),
        _box("Nơi thường trú / Place of residence:", 300, 525, 850, 548),
    ]

    layout = estimate_field_crop_layout(
        text_boxes,
        image_size=(1000, 630),
    )
    regions = layout["regions"]

    assert set(layout["labelAnchors"]) >= {
        "idNumber",
        "fullName",
        "dateOfBirth",
        "gender",
        "nationality",
        "placeOfOrigin",
        "placeOfResidence",
    }
    assert regions["idNumber"]["field"]["y2"] == regions["fullName"]["field"]["y1"]
    assert regions["fullName"]["field"]["y2"] == regions["dateOfBirth"]["field"]["y1"]
    assert regions["dateOfBirth"]["field"]["y2"] == regions["gender"]["field"]["y1"]
    assert regions["gender"]["field"]["y2"] == regions["placeOfOrigin"]["field"]["y1"]
    assert regions["placeOfOrigin"]["field"]["y2"] == regions["placeOfResidence"]["field"]["y1"]
    assert regions["gender"]["field"]["y1"] == regions["nationality"]["field"]["y1"]
    assert regions["gender"]["field"]["y2"] == regions["nationality"]["field"]["y2"]


def test_cropper_uses_label_layout_and_keeps_raw_variant_first(
    tmp_path: Path,
) -> None:
    card = np.full((630, 1000, 3), 210, dtype=np.uint8)
    layout = {
        "boundaryY": 540,
        "regions": {
            "fullName": {
                "field": {"x1": 270, "y1": 320, "x2": 975, "y2": 382},
                "value": {"x1": 270, "y1": 320, "x2": 985, "y2": 382},
            },
        },
    }
    result = CCCDFieldCropper().crop_fields(
        image=card,
        output_dir=str(tmp_path / "fields"),
        field_layout=layout,
    )

    assert result["fullName"]["box"] == [
        [270, 320],
        [975, 320],
        [975, 382],
        [270, 382],
    ]
    variants = result["fullName"]["variantImagePaths"]
    assert variants[0]["variant"] == "value_raw"
    assert variants[1]["variant"] == "value_processed"
    assert cv2.imread(result["portrait"]["rawImagePath"]).shape[:2] == (
        345,
        290,
    )


def test_full_card_reference_allows_one_confirming_field_ocr() -> None:
    assert FieldOCRService.can_stop_field_retries(
        field_name="fullName",
        candidates=[{"value": "ĐINH XUÂN HOÀNG"}],
        reference_value="ĐINH XUÂN HOÀNG",
    )
    assert FieldOCRService.can_stop_field_retries(
        field_name="placeOfOrigin",
        candidates=[{"value": "Lâm Trung Thủy, Đức Thọ, Hà Tĩnh"}],
        reference_value="Lam Trung Thuy, Duc Tho, Ha Tinh",
    )
    assert not FieldOCRService.can_stop_field_retries(
        field_name="placeOfOrigin",
        candidates=[{"value": "Lâm Trung Thủy, Đức Thọ, Hà Tĩnh"}],
        reference_value="Một địa chỉ khác, Hà Nội",
    )
