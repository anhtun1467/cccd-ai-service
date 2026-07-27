from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT_DIR),
    )


from app.modules.ocr.layout.box_utils import (
    filter_by_confidence,
    find_boxes_inside_rectangle,
    group_boxes_into_lines,
    merge_boxes,
    merge_same_line_boxes,
    prepare_layout_boxes,
    remove_duplicate_boxes,
    sort_left_to_right,
    sort_reading_order,
)
from app.modules.ocr.layout.text_box import (
    LayoutTextBox,
)


def create_box(
    text: str,
    left: float,
    top: float,
    right: float,
    bottom: float,
    confidence: float = 0.9,
) -> LayoutTextBox:
    return LayoutTextBox.from_rectangle(
        text=text,
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        confidence=confidence,
    )


def test_sort_left_to_right() -> None:
    boxes = [
        create_box(
            "Nam",
            503,
            475,
            579,
            511,
        ),
        create_box(
            "Gioi tinh / Sex:",
            318,
            478,
            493,
            510,
        ),
    ]

    result = sort_left_to_right(
        boxes
    )

    assert result[0].text == (
        "Gioi tinh / Sex:"
    )

    assert result[1].text == "Nam"


def test_group_into_lines() -> None:
    boxes = [
        create_box(
            "Gioi tinh / Sex:",
            318,
            478,
            493,
            510,
        ),
        create_box(
            "Nam",
            503,
            475,
            579,
            511,
        ),
        create_box(
            "Quoc tich / Nationality:",
            591,
            475,
            841,
            514,
        ),
        create_box(
            "Viet Nam",
            857,
            473,
            1001,
            513,
        ),
        create_box(
            "Que quan / Place of origin:",
            317,
            523,
            615,
            559,
        ),
    ]

    lines = group_boxes_into_lines(
        boxes
    )

    assert len(lines) == 2

    assert lines[0].text == (
        "Gioi tinh / Sex: Nam "
        "Quoc tich / Nationality: "
        "Viet Nam"
    )

    assert lines[1].text == (
        "Que quan / Place of origin:"
    )


def test_merge_boxes() -> None:
    boxes = [
        create_box(
            "NGUYEN",
            314,
            384,
            450,
            436,
            confidence=0.98,
        ),
        create_box(
            "HOANG NAM",
            460,
            384,
            739,
            436,
            confidence=0.97,
        ),
    ]

    merged = merge_boxes(
        boxes
    )

    assert merged.text == (
        "NGUYEN HOANG NAM"
    )

    assert merged.left == 314
    assert merged.right == 739
    assert merged.top == 384
    assert merged.bottom == 436


def test_merge_same_line_boxes() -> None:
    boxes = [
        create_box(
            "S6 / No:",
            317,
            309,
            421,
            345,
        ),
        create_box(
            "001095014159",
            424,
            301,
            780,
            355,
        ),
        create_box(
            "Ho va ten / Full name:",
            318,
            358,
            562,
            390,
        ),
    ]

    merged = merge_same_line_boxes(
        boxes
    )

    assert len(merged) == 2

    assert merged[0].text == (
        "S6 / No: 001095014159"
    )

    assert merged[1].text == (
        "Ho va ten / Full name:"
    )


def test_filter_confidence() -> None:
    boxes = [
        create_box(
            "Good",
            0,
            0,
            100,
            40,
            confidence=0.90,
        ),
        create_box(
            "Bad",
            0,
            50,
            100,
            90,
            confidence=0.01,
        ),
    ]

    result = filter_by_confidence(
        boxes,
        minimum_confidence=0.05,
    )

    assert len(result) == 1
    assert result[0].text == "Good"


def test_remove_duplicates() -> None:
    boxes = [
        create_box(
            "001095014159",
            424,
            301,
            780,
            355,
            confidence=0.99,
        ),
        create_box(
            "001095014159",
            426,
            302,
            779,
            354,
            confidence=0.75,
        ),
    ]

    result = remove_duplicate_boxes(
        boxes,
        minimum_iou=0.80,
    )

    assert len(result) == 1

    assert result[0].confidence == 0.99


def test_find_inside_rectangle() -> None:
    boxes = [
        create_box(
            "Nam",
            503,
            475,
            579,
            511,
        ),
        create_box(
            "Viet Nam",
            857,
            473,
            1001,
            513,
        ),
        create_box(
            "NGUYEN HOANG NAM",
            314,
            384,
            739,
            436,
        ),
    ]

    result = find_boxes_inside_rectangle(
        boxes=boxes,
        left=300,
        top=450,
        right=650,
        bottom=530,
    )

    assert len(result) == 1
    assert result[0].text == "Nam"


def test_prepare_layout_boxes() -> None:
    raw_items = [
        {
            "text": "Nam",
            "originalText": "Nam",
            "confidence": 0.99,
            "box": [
                [503, 475],
                [579, 475],
                [579, 511],
                [503, 511],
            ],
        },
        {
            "text": "Nam",
            "originalText": "Nam",
            "confidence": 0.50,
            "box": [
                [504, 476],
                [578, 476],
                [578, 510],
                [504, 510],
            ],
        },
        {
            "text": "Noise",
            "confidence": 0.01,
            "box": [
                [0, 0],
                [20, 0],
                [20, 20],
                [0, 20],
            ],
        },
    ]

    result = prepare_layout_boxes(
        raw_items,
        minimum_confidence=0.05,
        remove_duplicates=True,
    )

    assert len(result) == 1
    assert result[0].text == "Nam"
    assert result[0].confidence == 0.99


def test_reading_order() -> None:
    boxes = [
        create_box(
            "Second line",
            100,
            200,
            300,
            240,
        ),
        create_box(
            "Right",
            300,
            100,
            400,
            140,
        ),
        create_box(
            "Left",
            100,
            100,
            200,
            140,
        ),
    ]

    result = sort_reading_order(
        boxes
    )

    assert result[0].text == "Left"
    assert result[1].text == "Right"
    assert result[2].text == (
        "Second line"
    )


def main() -> None:
    tests = (
        test_sort_left_to_right,
        test_group_into_lines,
        test_merge_boxes,
        test_merge_same_line_boxes,
        test_filter_confidence,
        test_remove_duplicates,
        test_find_inside_rectangle,
        test_prepare_layout_boxes,
        test_reading_order,
    )

    passed = 0

    for test_function in tests:
        try:
            test_function()

            print(
                f"[PASS] "
                f"{test_function.__name__}"
            )

            passed += 1

        except Exception as error:
            print(
                f"[FAIL] "
                f"{test_function.__name__}: "
                f"{error}"
            )

    print("-" * 60)

    print(
        f"Kết quả: "
        f"{passed}/{len(tests)} "
        f"test thành công"
    )

    if passed != len(tests):
        raise AssertionError(
            "Có test Box Utils chưa đạt"
        )


if __name__ == "__main__":
    main()
