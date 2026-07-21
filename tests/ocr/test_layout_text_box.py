from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from app.modules.ocr.layout.text_box import (
    LayoutTextBox,
    convert_to_layout_boxes,
)


def test_from_dict() -> None:
    data = {
        "text": "NGUYEN HOANG NAM",
        "originalText": "NGUYEN HOANG NAM",
        "confidence": 0.9783,
        "box": [
            [314, 384],
            [739, 384],
            [739, 436],
            [314, 436],
        ],
    }

    text_box = LayoutTextBox.from_dict(data)

    assert text_box.text == "NGUYEN HOANG NAM"
    assert text_box.left == 314
    assert text_box.top == 384
    assert text_box.right == 739
    assert text_box.bottom == 436
    assert text_box.width == 425
    assert text_box.height == 52
    assert text_box.center == (526.5, 410.0)


def test_from_rectangle() -> None:
    text_box = LayoutTextBox.from_rectangle(
        text="24/03/1995",
        left=611,
        top=439,
        right=789,
        bottom=477,
        confidence=0.5039,
    )

    assert text_box.text == "24/03/1995"
    assert text_box.width == 178
    assert text_box.height == 38
    assert text_box.area == 6764


def test_order_points() -> None:
    unordered_box = [
        [739, 436],
        [314, 384],
        [314, 436],
        [739, 384],
    ]

    text_box = LayoutTextBox.from_dict(
        {
            "text": "NGUYEN HOANG NAM",
            "confidence": 0.98,
            "box": unordered_box,
        }
    )

    assert text_box.box == (
        (314.0, 384.0),
        (739.0, 384.0),
        (739.0, 436.0),
        (314.0, 436.0),
    )


def test_convert_list() -> None:
    items = [
        {
            "text": "Nam",
            "confidence": 0.99,
            "box": [
                [503, 475],
                [579, 475],
                [579, 511],
                [503, 511],
            ],
        },
        {
            "text": "",
            "confidence": 0.5,
            "box": [
                [0, 0],
                [10, 0],
                [10, 10],
                [0, 10],
            ],
        },
        {
            "text": "Box lỗi",
            "confidence": 0.5,
            "box": [],
        },
    ]

    results = convert_to_layout_boxes(items)

    assert len(results) == 1
    assert results[0].text == "Nam"


def main() -> None:
    tests = (
        test_from_dict,
        test_from_rectangle,
        test_order_points,
        test_convert_list,
    )

    passed = 0

    for test_function in tests:
        try:
            test_function()
            print(
                f"[PASS] {test_function.__name__}"
            )
            passed += 1
        except Exception as error:
            print(
                f"[FAIL] {test_function.__name__}: "
                f"{error}"
            )

    print("-" * 60)
    print(
        f"Kết quả: {passed}/{len(tests)} test thành công"
    )

    if passed != len(tests):
        raise AssertionError(
            "Có test LayoutTextBox chưa đạt"
        )


if __name__ == "__main__":
    main()