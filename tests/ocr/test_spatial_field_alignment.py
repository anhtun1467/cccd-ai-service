from app.modules.ocr.result_fuser import (
    estimate_address_crop_layout,
    estimate_layout_y_offset,
    fuse_ocr_data,
    recover_spatial_address,
)


def _box(text: str, left: float, top: float, right: float, bottom: float):
    return {
        "text": text,
        "confidence": 0.9,
        "box": [
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom],
        ],
    }


def test_layout_offset_tracks_shifted_identifier_row() -> None:
    lower = [_box("001205016122", 320, 312, 620, 342)]
    upper = [_box("001205016122", 320, 228, 620, 258)]

    assert estimate_layout_y_offset(lower, (1000, 630)) == 42.0
    assert estimate_layout_y_offset(upper, (1000, 630)) == -42.0


def test_layout_offset_ignores_twelve_letter_noise() -> None:
    boxes = [_box("OOOOIIIIllll", 320, 312, 620, 342)]

    assert estimate_layout_y_offset(boxes, (1000, 630)) == 0.0


def test_residence_label_defines_shared_address_crop_boundary() -> None:
    boxes = [
        _box("001205016122", 320, 270, 620, 300),
        _box("Quê quán / Place of origin", 300, 470, 590, 495),
        _box(
            "Nơi thường trú / Place of residence",
            300,
            526,
            690,
            551,
        ),
    ]

    layout = estimate_address_crop_layout(boxes, (1000, 630))

    assert layout["source"] == "residence_label"
    assert layout["boundaryY"] == 524.0
    assert layout["labelAnchors"]["placeOfOrigin"]["bottom"] == 495.0
    assert (
        layout["labelAnchors"]["placeOfResidence"]["top"]
        == 526.0
    )


def test_origin_is_rebuilt_by_x_position_instead_of_ocr_order() -> None:
    boxes = [
        _box("Quê quán / Place of origin", 35, 470, 300, 495),
        # Deliberately pass the pieces in the wrong reading order.
        _box("Bắc Giang", 755, 505, 930, 530),
        _box("Lê Lợi,", 305, 505, 410, 530),
        _box("Thành phố Bắc Giang,", 420, 505, 745, 530),
    ]

    assert recover_spatial_address(
        boxes,
        "placeOfOrigin",
        (1000, 630),
    ) == "Lê Lợi, Thành phố Bắc Giang, Bắc Giang"


def test_residence_excludes_expiry_column_on_same_visual_row() -> None:
    boxes = [
        _box("Nơi thường trú / Place of residence", 25, 520, 390, 550),
        _box("18", 400, 520, 435, 550),
        _box("Có giá trị đến / Date of expiry", 25, 565, 245, 590),
        _box("14/03/2028", 85, 592, 235, 618),
        _box(
            "Nguyên Hồng, Tân Sơn, Thành phố Thanh Hóa, Thanh Hóa",
            285,
            565,
            950,
            590,
        ),
    ]

    assert recover_spatial_address(
        boxes,
        "placeOfResidence",
        (1000, 630),
    ) == (
        "18 Nguyên Hồng, Tân Sơn, Thành phố Thanh Hóa, Thanh Hóa"
    )


def test_cross_field_crop_leak_is_not_reported_as_valid_residence() -> None:
    result, sources = fuse_ocr_data(
        full_card_data={
            "placeOfOrigin": (
                "Thị trấn Thanh Lãng, Bình Xuyên, Vĩnh Phúc"
            ),
        },
        field_data={
            "placeOfResidence": "Xuyên, Vĩnh, Phú Yên",
        },
        raw_text=[],
    )

    assert result["placeOfResidence"] is None
    assert sources["placeOfResidence"] == "NOT_FOUND"
