from __future__ import annotations

import math
from dataclasses import dataclass

from app.modules.ocr.layout.text_box import (
    LayoutTextBox,
)


@dataclass(frozen=True, slots=True)
class BoxDistance:
    """
    Thông tin khoảng cách giữa hai OCR box.
    """

    horizontal: float
    vertical: float
    center: float

    def to_dict(self) -> dict[str, float]:
        return {
            "horizontal": round(
                self.horizontal,
                4,
            ),
            "vertical": round(
                self.vertical,
                4,
            ),
            "center": round(
                self.center,
                4,
            ),
        }


def safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    """
    Thực hiện phép chia an toàn.

    Trả về 0.0 nếu mẫu số không hợp lệ.
    """

    if denominator <= 0:
        return 0.0

    return numerator / denominator


def horizontal_overlap(
    first: LayoutTextBox,
    second: LayoutTextBox,
) -> float:
    """
    Độ dài chồng nhau theo trục X.
    """

    return max(
        0.0,
        min(
            first.right,
            second.right,
        )
        - max(
            first.left,
            second.left,
        ),
    )


def vertical_overlap(
    first: LayoutTextBox,
    second: LayoutTextBox,
) -> float:
    """
    Độ dài chồng nhau theo trục Y.
    """

    return max(
        0.0,
        min(
            first.bottom,
            second.bottom,
        )
        - max(
            first.top,
            second.top,
        ),
    )


def horizontal_overlap_ratio(
    first: LayoutTextBox,
    second: LayoutTextBox,
    relative_to: str = "minimum",
) -> float:
    """
    Tính tỷ lệ chồng theo trục X.

    relative_to:
        minimum:
            Chia cho chiều rộng nhỏ hơn.

        first:
            Chia cho chiều rộng box thứ nhất.

        second:
            Chia cho chiều rộng box thứ hai.

        union:
            Chia cho tổng vùng bao phủ theo X.
    """

    overlap = horizontal_overlap(
        first,
        second,
    )

    if relative_to == "first":
        denominator = first.width

    elif relative_to == "second":
        denominator = second.width

    elif relative_to == "union":
        denominator = (
            max(
                first.right,
                second.right,
            )
            - min(
                first.left,
                second.left,
            )
        )

    else:
        denominator = min(
            first.width,
            second.width,
        )

    return safe_ratio(
        overlap,
        denominator,
    )


def vertical_overlap_ratio(
    first: LayoutTextBox,
    second: LayoutTextBox,
    relative_to: str = "minimum",
) -> float:
    """
    Tính tỷ lệ chồng theo trục Y.
    """

    overlap = vertical_overlap(
        first,
        second,
    )

    if relative_to == "first":
        denominator = first.height

    elif relative_to == "second":
        denominator = second.height

    elif relative_to == "union":
        denominator = (
            max(
                first.bottom,
                second.bottom,
            )
            - min(
                first.top,
                second.top,
            )
        )

    else:
        denominator = min(
            first.height,
            second.height,
        )

    return safe_ratio(
        overlap,
        denominator,
    )


def center_distance(
    first: LayoutTextBox,
    second: LayoutTextBox,
) -> float:
    """
    Khoảng cách Euclidean giữa tâm hai box.
    """

    delta_x = (
        first.center_x
        - second.center_x
    )

    delta_y = (
        first.center_y
        - second.center_y
    )

    return math.hypot(
        delta_x,
        delta_y,
    )


def horizontal_gap(
    first: LayoutTextBox,
    second: LayoutTextBox,
) -> float:
    """
    Khoảng trống theo trục X giữa hai box.

    Nếu hai box chồng nhau theo X thì trả về 0.
    """

    if first.right < second.left:
        return second.left - first.right

    if second.right < first.left:
        return first.left - second.right

    return 0.0


def vertical_gap(
    first: LayoutTextBox,
    second: LayoutTextBox,
) -> float:
    """
    Khoảng trống theo trục Y giữa hai box.

    Nếu hai box chồng nhau theo Y thì trả về 0.
    """

    if first.bottom < second.top:
        return second.top - first.bottom

    if second.bottom < first.top:
        return first.top - second.bottom

    return 0.0


def calculate_box_distance(
    first: LayoutTextBox,
    second: LayoutTextBox,
) -> BoxDistance:
    """
    Tính đầy đủ khoảng cách giữa hai box.
    """

    return BoxDistance(
        horizontal=horizontal_gap(
            first,
            second,
        ),
        vertical=vertical_gap(
            first,
            second,
        ),
        center=center_distance(
            first,
            second,
        ),
    )


def is_same_line(
    first: LayoutTextBox,
    second: LayoutTextBox,
    minimum_vertical_overlap: float = 0.45,
    maximum_center_y_difference_ratio: float = 0.65,
) -> bool:
    """
    Kiểm tra hai box có nằm trên cùng một dòng không.

    Dùng kết hợp:
    - Tỷ lệ chồng theo trục Y.
    - Độ lệch tâm Y so với chiều cao trung bình.
    """

    overlap_ratio = vertical_overlap_ratio(
        first,
        second,
        relative_to="minimum",
    )

    average_height = (
        first.height + second.height
    ) / 2.0

    center_y_difference = abs(
        first.center_y
        - second.center_y
    )

    center_ratio = safe_ratio(
        center_y_difference,
        average_height,
    )

    return (
        overlap_ratio
        >= minimum_vertical_overlap
        or center_ratio
        <= maximum_center_y_difference_ratio
    )


def is_to_right_of(
    candidate: LayoutTextBox,
    anchor: LayoutTextBox,
    minimum_vertical_overlap: float = 0.30,
    allow_small_left_overlap: bool = True,
) -> bool:
    """
    Kiểm tra candidate có nằm bên phải anchor không.
    """

    if allow_small_left_overlap:
        right_condition = (
            candidate.center_x
            > anchor.center_x
        )
    else:
        right_condition = (
            candidate.left
            >= anchor.right
        )

    if not right_condition:
        return False

    overlap_ratio = vertical_overlap_ratio(
        candidate,
        anchor,
        relative_to="minimum",
    )

    return (
        overlap_ratio
        >= minimum_vertical_overlap
        or is_same_line(
            candidate,
            anchor,
        )
    )


def is_below(
    candidate: LayoutTextBox,
    anchor: LayoutTextBox,
    minimum_horizontal_overlap: float = 0.20,
    allow_small_top_overlap: bool = True,
) -> bool:
    """
    Kiểm tra candidate có nằm bên dưới anchor không.
    """

    if allow_small_top_overlap:
        below_condition = (
            candidate.center_y
            > anchor.center_y
        )
    else:
        below_condition = (
            candidate.top
            >= anchor.bottom
        )

    if not below_condition:
        return False

    overlap_ratio = horizontal_overlap_ratio(
        candidate,
        anchor,
        relative_to="minimum",
    )

    return (
        overlap_ratio
        >= minimum_horizontal_overlap
    )


def is_above(
    candidate: LayoutTextBox,
    anchor: LayoutTextBox,
    minimum_horizontal_overlap: float = 0.20,
) -> bool:
    """
    Kiểm tra candidate có nằm phía trên anchor không.
    """

    return is_below(
        candidate=anchor,
        anchor=candidate,
        minimum_horizontal_overlap=(
            minimum_horizontal_overlap
        ),
    )


def is_to_left_of(
    candidate: LayoutTextBox,
    anchor: LayoutTextBox,
    minimum_vertical_overlap: float = 0.30,
) -> bool:
    """
    Kiểm tra candidate có nằm bên trái anchor không.
    """

    return is_to_right_of(
        candidate=anchor,
        anchor=candidate,
        minimum_vertical_overlap=(
            minimum_vertical_overlap
        ),
    )


def contains_point(
    text_box: LayoutTextBox,
    x: float,
    y: float,
) -> bool:
    """
    Kiểm tra một điểm có nằm trong hình chữ nhật bao ngoài hay không.
    """

    return (
        text_box.left <= x <= text_box.right
        and text_box.top <= y <= text_box.bottom
    )


def contains_box(
    outer: LayoutTextBox,
    inner: LayoutTextBox,
    tolerance: float = 0.0,
) -> bool:
    """
    Kiểm tra outer có bao chứa inner không.
    """

    return (
        inner.left
        >= outer.left - tolerance
        and inner.right
        <= outer.right + tolerance
        and inner.top
        >= outer.top - tolerance
        and inner.bottom
        <= outer.bottom + tolerance
    )


def intersection_area(
    first: LayoutTextBox,
    second: LayoutTextBox,
) -> float:
    """
    Diện tích giao nhau của hai hình chữ nhật bao ngoài.
    """

    return (
        horizontal_overlap(
            first,
            second,
        )
        * vertical_overlap(
            first,
            second,
        )
    )


def intersection_over_union(
    first: LayoutTextBox,
    second: LayoutTextBox,
) -> float:
    """
    Tính Intersection over Union — IoU.
    """

    intersection = intersection_area(
        first,
        second,
    )

    union = (
        first.area
        + second.area
        - intersection
    )

    return safe_ratio(
        intersection,
        union,
    )


def normalized_horizontal_gap(
    first: LayoutTextBox,
    second: LayoutTextBox,
) -> float:
    """
    Khoảng cách X được chuẩn hóa theo chiều cao trung bình.

    Giá trị này hữu ích vì khoảng cách chữ thường
    phụ thuộc vào kích thước dòng.
    """

    average_height = (
        first.height + second.height
    ) / 2.0

    return safe_ratio(
        horizontal_gap(
            first,
            second,
        ),
        average_height,
    )


def normalized_vertical_gap(
    first: LayoutTextBox,
    second: LayoutTextBox,
) -> float:
    """
    Khoảng cách Y được chuẩn hóa theo chiều cao trung bình.
    """

    average_height = (
        first.height + second.height
    ) / 2.0

    return safe_ratio(
        vertical_gap(
            first,
            second,
        ),
        average_height,
    )


def score_right_candidate(
    anchor: LayoutTextBox,
    candidate: LayoutTextBox,
    maximum_gap_ratio: float = 8.0,
) -> float:
    """
    Chấm điểm candidate nằm bên phải anchor.

    Điểm càng cao thì candidate càng phù hợp.

    Các yếu tố:
    - Cùng dòng.
    - Chồng theo trục Y.
    - Khoảng cách ngang nhỏ.
    - Confidence OCR cao.
    """

    if not is_to_right_of(
        candidate,
        anchor,
    ):
        return float("-inf")

    gap_ratio = normalized_horizontal_gap(
        anchor,
        candidate,
    )

    if gap_ratio > maximum_gap_ratio:
        return float("-inf")

    same_line_score = (
        1.0
        if is_same_line(
            anchor,
            candidate,
        )
        else 0.0
    )

    overlap_score = vertical_overlap_ratio(
        anchor,
        candidate,
        relative_to="minimum",
    )

    distance_penalty = min(
        gap_ratio / maximum_gap_ratio,
        1.0,
    )

    confidence_score = (
        candidate.confidence
    )

    return (
        same_line_score * 4.0
        + overlap_score * 3.0
        + confidence_score * 2.0
        - distance_penalty * 3.0
    )


def score_below_candidate(
    anchor: LayoutTextBox,
    candidate: LayoutTextBox,
    maximum_gap_ratio: float = 5.0,
) -> float:
    """
    Chấm điểm candidate nằm phía dưới anchor.

    Thường dùng để tìm:
    - Họ tên phía dưới nhãn.
    - Quê quán phía dưới nhãn.
    - Nơi thường trú phía dưới nhãn.
    """

    if not is_below(
        candidate,
        anchor,
    ):
        return float("-inf")

    gap_ratio = normalized_vertical_gap(
        anchor,
        candidate,
    )

    if gap_ratio > maximum_gap_ratio:
        return float("-inf")

    overlap_score = horizontal_overlap_ratio(
        anchor,
        candidate,
        relative_to="minimum",
    )

    horizontal_alignment = max(
        0.0,
        1.0
        - safe_ratio(
            abs(
                candidate.left
                - anchor.left
            ),
            max(
                anchor.width,
                candidate.width,
            ),
        ),
    )

    distance_penalty = min(
        gap_ratio / maximum_gap_ratio,
        1.0,
    )

    confidence_score = (
        candidate.confidence
    )

    return (
        overlap_score * 3.0
        + horizontal_alignment * 2.0
        + confidence_score * 2.0
        - distance_penalty * 3.0
    )


def reading_order_key(
    text_box: LayoutTextBox,
) -> tuple[float, float]:
    """
    Khóa sắp xếp theo thứ tự đọc:
    trên xuống dưới, trái sang phải.
    """

    return (
        text_box.center_y,
        text_box.left,
    )
