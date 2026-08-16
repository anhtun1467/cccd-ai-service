from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.modules.ocr.field_cropper import CCCDFieldCropper


def test_origin_and_residence_regions_never_overlap() -> None:
    cropper = CCCDFieldCropper()

    for offset in (-55.0, 0.0, 55.0):
        for region_kind, regions in (
            ("field", cropper.FIELD_REGIONS),
            ("value", cropper.VALUE_REGIONS),
        ):
            origin = cropper.resolve_region(
                "placeOfOrigin",
                regions["placeOfOrigin"],
                offset,
                region_kind=region_kind,
            )
            residence = cropper.resolve_region(
                "placeOfResidence",
                regions["placeOfResidence"],
                offset,
                region_kind=region_kind,
            )

            assert origin.y2 == residence.y1
            assert origin.y1 < origin.y2
            assert residence.y1 < residence.y2


def test_label_anchor_moves_both_address_crops_to_one_boundary(
    tmp_path: Path,
) -> None:
    boundary = 548
    card = np.full((630, 1000, 3), 225, dtype=np.uint8)
    card[455:boundary, 300:995] = (40, 190, 40)
    card[boundary:630, 300:995] = (190, 40, 40)
    cropper = CCCDFieldCropper()

    result = cropper.crop_fields(
        image=card,
        output_dir=str(tmp_path / "fields"),
        address_layout={
            "boundaryY": boundary,
            "source": "residence_label",
        },
    )

    origin_box = result["placeOfOrigin"]["box"]
    residence_box = result["placeOfResidence"]["box"]
    assert origin_box[2][1] == boundary
    assert residence_box[0][1] == boundary

    origin = cv2.imread(result["placeOfOrigin"]["rawImagePath"])
    residence = cv2.imread(result["placeOfResidence"]["rawImagePath"])
    assert origin is not None
    assert residence is not None
    assert float(np.mean(origin[:, :, 1])) > float(np.mean(origin[:, :, 0]))
    assert float(np.mean(residence[:, :, 0])) > float(
        np.mean(residence[:, :, 1])
    )

