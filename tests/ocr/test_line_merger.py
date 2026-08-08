from app.modules.ocr.line_merger import OCRLineMerger


def _box(text, left, top, right, bottom):
    return {
        "text": text,
        "confidence": 0.8,
        "box": [
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom],
        ],
    }


def test_expiry_column_is_not_merged_into_residence_lines():
    merger = OCRLineMerger(
        vertical_tolerance_ratio=0.25,
        maximum_horizontal_gap_ratio=1.8,
    )
    boxes = [
        _box("Cogla", 54, 592, 122, 624),
        _box("Noi thưong tú / Place of", 333, 573, 631, 609),
        _box("'residonceThôn 48", 622, 568, 877, 629),
        _box("Dalctokoro", 52, 616, 160, 642),
        _box("1/01/2028", 187, 599, 313, 635),
        _box("Ea Hiao, Ea Hleo, Đák Lák", 323, 617, 741, 661),
        _box("àoen", 112, 584, 196, 630),
    ]

    lines = [item["text"] for item in merger.merge(boxes)]

    assert "Noi thuong tru / Place of residence Thôn 48" in lines
    assert "Ea Hiao, Ea Hleo, Đák Lák" in lines
    assert not any(
        "1/01/2028" in line and "Ea Hiao" in line
        for line in lines
    )


def test_tall_box_cannot_bridge_two_address_rows():
    merger = OCRLineMerger(
        vertical_tolerance_ratio=0.30,
        maximum_horizontal_gap_ratio=2.2,
        minimum_vertical_overlap_ratio=0.40,
    )
    boxes = [
        _box("Nơi thường trú", 0, 10, 100, 30),
        # Box cao do EasyOCR bắt cả họa tiết; thuật toán union cũ làm
        # chiều cao row phình ra và kéo dòng dưới vào cùng row.
        _box("Place of residence", 105, 15, 250, 60),
        _box("Tân Sơn, Thành phố Thanh Hóa", 255, 40, 520, 60),
    ]

    lines = [item["text"] for item in merger.merge(boxes)]

    assert "Tân Sơn, Thành phố Thanh Hóa" in lines
    assert not any(
        "Nơi thường trú" in line and "Tân Sơn" in line
        for line in lines
    )
