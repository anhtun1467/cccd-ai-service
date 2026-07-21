from __future__ import annotations

import math
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from app.modules.ocr.layout.geometry import (
    calculate_box_distance,
    horizontal_gap,
    horizontal_overlap_ratio,
    intersection_over_union,
    is_below,
    is_same_line,
    is_to_right_of,
    normalized_horizontal_gap,
    score_below_candidate,
    score_right_candidate,
    vertical_gap,
    vertical_overlap_ratio,
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


def test_same_line() -> None:
    label = create_box(
        text="Gioi tinh / Sex:",
        left=318,
        top=478,
        right=493,
        bottom=510,
    )

    value = create_box(
        text="Nam",
        left=503,
        top=475,
        right=579,
        bottom=511,
    )

    assert is_same_line(
        label,
        value,
    )

    assert is_to_right_of(
        value,
        label,
    )


def test_box_below() -> None:
    label = create_box(
        text="Ho va ten / Full name:",
        left=318,
        top=358,
        right=562,
        bottom=390,
    )

    value = create_box(
        text="NGUYEN HOANG NAM",
        left=314,
        top=384,
        right=739,
        bottom=436,
    )

    assert is_below(
        value,
        label,
    )

    assert not is_to_right_of(
        value,
        label,
        allow_small_left_overlap=False,
    )


def test_gaps() -> None:
    left_box = create_box(
        text="Label",
        left=100,
        top=100,
        right=200,
        bottom=140,
    )

    right_box = create_box(
        text="Value",
        left=220,
        top=100,
        right=320,
        bottom=140,
    )

    bottom_box = create_box(
        text="Bottom",
        left=100,
        top=160,
        right=250,
        bottom=200,
    )

    assert horizontal_gap(
        left_box,
        right_box,
    ) == 20

    assert vertical_gap(
        left_box,
        bottom_box,
    ) == 20

    assert vertical_gap(
        left_box,
        right_box,
    ) == 0


def test_overlap_ratios() -> None:
    first = create_box(
        text="First",
        left=0,
        top=0,
        right=100,
        bottom=50,
    )

    second = create_box(
        text="Second",
        left=50,
        top=10,
        right=150,
        bottom=40,
    )

    x_ratio = horizontal_overlap_ratio(
        first,
        second,
    )

    y_ratio = vertical_overlap_ratio(
        first,
        second,
    )

    assert math.isclose(
        x_ratio,
        0.5,
    )

    assert math.isclose(
        y_ratio,
        1.0,
    )


def test_distance() -> None:
    first = create_box(
        text="First",
        left=0,
        top=0,
        right=100,
        bottom=50,
    )

    second = create_box(
        text="Second",
        left=120,
        top=0,
        right=220,
        bottom=50,
    )

    distance = calculate_box_distance(
        first,
        second,
    )

    assert distance.horizontal == 20
    assert distance.vertical == 0
    assert distance.center == 120


def test_iou() -> None:
    first = create_box(
        text="First",
        left=0,
        top=0,
        right=100,
        bottom=100,
    )

    second = create_box(
        text="Second",
        left=50,
        top=50,
        right=150,
        bottom=150,
    )

    result = intersection_over_union(
        first,
        second,
    )

    expected = 2500 / 17500

    assert math.isclose(
        result,
        expected,
        rel_tol=1e-6,
    )


def test_right_candidate_score() -> None:
    label = create_box(
        text="Gioi tinh / Sex:",
        left=318,
        top=478,
        right=493,
        bottom=510,
        confidence=0.82,
    )

    correct_value = create_box(
        text="Nam",
        left=503,
        top=475,
        right=579,
        bottom=511,
        confidence=0.99,
    )

    wrong_value = create_box(
        text="Viet Nam",
        left=857,
        top=473,
        right=1001,
        bottom=513,
        confidence=0.12,
    )

    correct_score = score_right_candidate(
        label,
        correct_value,
    )

    wrong_score = score_right_candidate(
        label,
        wrong_value,
    )

    assert correct_score > wrong_score


def test_below_candidate_score() -> None:
    label = create_box(
        text="Ho va ten / Full name:",
        left=318,
        top=358,
        right=562,
        bottom=390,
    )

    correct_name = create_box(
        text="NGUYEN HOANG NAM",
        left=314,
        top=384,
        right=739,
        bottom=436,
        confidence=0.98,
    )

    distant_box = create_box(
        text="Bach Dang",
        left=317,
        top=555,
        right=493,
        bottom=591,
        confidence=0.79,
    )

    correct_score = score_below_candidate(
        label,
        correct_name,
    )

    distant_score = score_below_candidate(
        label,
        distant_box,
    )

    assert correct_score > distant_score


def test_normalized_gap() -> None:
    first = create_box(
        text="First",
        left=0,
        top=0,
        right=100,
        bottom=40,
    )

    second = create_box(
        text="Second",
        left=120,
        top=0,
        right=220,
        bottom=40,
    )

    result = normalized_horizontal_gap(
        first,
        second,
    )

    assert math.isclose(
        result,
        0.5,
    )


def main() -> None:
    tests = (
        test_same_line,
        test_box_below,
        test_gaps,
        test_overlap_ratios,
        test_distance,
        test_iou,
        test_right_candidate_score,
        test_below_candidate_score,
        test_normalized_gap,
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
            "Có test Geometry Engine chưa đạt"
        )


if __name__ == "__main__":
    main()