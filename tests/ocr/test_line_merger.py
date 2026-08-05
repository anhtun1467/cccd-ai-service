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
