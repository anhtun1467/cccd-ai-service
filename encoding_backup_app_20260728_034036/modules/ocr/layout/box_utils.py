from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from app.modules.ocr.layout.geometry import (
    intersection_over_union,
    is_same_line,
    reading_order_key,
)
from app.modules.ocr.layout.text_box import (
    LayoutTextBox,
    convert_to_layout_boxes,
)


@dataclass(frozen=True, slots=True)
class TextLine:
    """
    Đại diện cho một dòng văn bản gồm một hoặc nhiều OCR box.
    """

    boxes: tuple[LayoutTextBox, ...]

    @property
    def text(self) -> str:
        """
        Nội dung hoàn chỉnh của dòng.
        """

        return " ".join(
            box.text.strip()
            for box in self.boxes
            if box.text.strip()
        ).strip()

    @property
    def original_text(self) -> str:
        """
        Nội dung OCR gốc của dòng.
        """

        return " ".join(
            box.original_text.strip()
            for box in self.boxes
            if box.original_text.strip()
        ).strip()

    @property
    def confidence(self) -> float:
        """
        Confidence trung bình có trọng số theo diện tích box.
        """

        if not self.boxes:
            return 0.0

        total_weight = 0.0
        weighted_score = 0.0

        for box in self.boxes:
            weight = max(
                box.area,
                1.0,
            )

            total_weight += weight
            weighted_score += (
                box.confidence * weight
            )

        if total_weight <= 0:
            return 0.0

        return weighted_score / total_weight

    @property
    def left(self) -> float:
        return min(
            box.left
            for box in self.boxes
        )

    @property
    def right(self) -> float:
        return max(
            box.right
            for box in self.boxes
        )

    @property
    def top(self) -> float:
        return min(
            box.top
            for box in self.boxes
        )

    @property
    def bottom(self) -> float:
        return max(
            box.bottom
            for box in self.boxes
        )

    @property
    def width(self) -> float:
        return max(
            0.0,
            self.right - self.left,
        )

    @property
    def height(self) -> float:
        return max(
            0.0,
            self.bottom - self.top,
        )

    @property
    def center_x(self) -> float:
        return (
            self.left + self.right
        ) / 2.0

    @property
    def center_y(self) -> float:
        return (
            self.top + self.bottom
        ) / 2.0

    def to_layout_box(self) -> LayoutTextBox:
        """
        Gộp toàn bộ box trong dòng thành một LayoutTextBox.
        """

        return LayoutTextBox.from_rectangle(
            text=self.text,
            original_text=self.original_text,
            left=self.left,
            top=self.top,
            right=self.right,
            bottom=self.bottom,
            confidence=self.confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Chuyển dòng văn bản thành dictionary JSON-safe.
        """

        return {
            "text": self.text,
            "originalText": self.original_text,
            "confidence": round(
                self.confidence,
                4,
            ),
            "boxCount": len(
                self.boxes
            ),
            "box": [
                [self.left, self.top],
                [self.right, self.top],
                [self.right, self.bottom],
                [self.left, self.bottom],
            ],
            "sourceBoxes": [
                box.to_dict()
                for box in self.boxes
            ],
        }


def sort_left_to_right(
    boxes: Iterable[LayoutTextBox],
) -> list[LayoutTextBox]:
    """
    Sắp xếp box từ trái sang phải.
    """

    return sorted(
        boxes,
        key=lambda box: (
            box.left,
            box.center_y,
        ),
    )


def sort_top_to_bottom(
    boxes: Iterable[LayoutTextBox],
) -> list[LayoutTextBox]:
    """
    Sắp xếp box từ trên xuống dưới.
    """

    return sorted(
        boxes,
        key=lambda box: (
            box.top,
            box.left,
        ),
    )


def sort_reading_order(
    boxes: Iterable[LayoutTextBox],
) -> list[LayoutTextBox]:
    """
    Sắp xếp theo thứ tự đọc:
    trên xuống dưới, trái sang phải.
    """

    return sorted(
        boxes,
        key=reading_order_key,
    )


def filter_by_confidence(
    boxes: Iterable[LayoutTextBox],
    minimum_confidence: float = 0.05,
) -> list[LayoutTextBox]:
    """
    Loại bỏ box có confidence quá thấp.
    """

    return [
        box
        for box in boxes
        if box.confidence
        >= minimum_confidence
    ]


def filter_by_minimum_size(
    boxes: Iterable[LayoutTextBox],
    minimum_width: float = 2.0,
    minimum_height: float = 2.0,
) -> list[LayoutTextBox]:
    """
    Loại bỏ box quá nhỏ hoặc không hợp lệ.
    """

    return [
        box
        for box in boxes
        if (
            box.width >= minimum_width
            and box.height >= minimum_height
        )
    ]


def remove_duplicate_boxes(
    boxes: Iterable[LayoutTextBox],
    minimum_iou: float = 0.80,
    compare_text: bool = True,
) -> list[LayoutTextBox]:
    """
    Loại bỏ OCR box trùng lặp.

    Khi hai box chồng nhau mạnh:
    - Nếu compare_text=True thì chỉ coi là trùng khi text gần giống.
    - Giữ lại box có confidence cao hơn.
    """

    ordered = sorted(
        boxes,
        key=lambda box: (
            box.confidence,
            box.area,
        ),
        reverse=True,
    )

    accepted: list[LayoutTextBox] = []

    for candidate in ordered:
        is_duplicate = False

        for existing in accepted:
            iou = intersection_over_union(
                candidate,
                existing,
            )

            if iou < minimum_iou:
                continue

            if compare_text:
                candidate_text = (
                    normalize_comparison_text(
                        candidate.text
                    )
                )

                existing_text = (
                    normalize_comparison_text(
                        existing.text
                    )
                )

                if (
                    candidate_text
                    != existing_text
                ):
                    continue

            is_duplicate = True
            break

        if not is_duplicate:
            accepted.append(
                candidate
            )

    return sort_reading_order(
        accepted
    )


def normalize_comparison_text(
    text: str,
) -> str:
    """
    Chuẩn hóa text để so sánh box trùng.
    """

    return "".join(
        character.lower()
        for character in text.strip()
        if character.isalnum()
    )


def group_boxes_into_lines(
    boxes: Iterable[LayoutTextBox],
    minimum_vertical_overlap: float = 0.35,
    maximum_center_y_difference_ratio: float = 0.80,
) -> list[TextLine]:
    """
    Nhóm các OCR box vào cùng dòng.

    Mỗi box được đưa vào dòng có độ lệch tâm Y nhỏ nhất
    nếu đáp ứng điều kiện cùng dòng.
    """

    ordered_boxes = sort_top_to_bottom(
        boxes
    )

    line_groups: list[
        list[LayoutTextBox]
    ] = []

    for candidate in ordered_boxes:
        best_line_index: int | None = None
        best_difference = float("inf")

        for index, line_boxes in enumerate(
            line_groups
        ):
            representative = (
                merge_boxes(
                    line_boxes
                )
            )

            same_line = is_same_line(
                representative,
                candidate,
                minimum_vertical_overlap=(
                    minimum_vertical_overlap
                ),
                maximum_center_y_difference_ratio=(
                    maximum_center_y_difference_ratio
                ),
            )

            if not same_line:
                continue

            difference = abs(
                representative.center_y
                - candidate.center_y
            )

            if difference < best_difference:
                best_difference = difference
                best_line_index = index

        if best_line_index is None:
            line_groups.append(
                [candidate]
            )
        else:
            line_groups[
                best_line_index
            ].append(candidate)

    lines: list[TextLine] = []

    for line_boxes in line_groups:
        ordered_line_boxes = (
            sort_left_to_right(
                line_boxes
            )
        )

        lines.append(
            TextLine(
                boxes=tuple(
                    ordered_line_boxes
                )
            )
        )

    return sorted(
        lines,
        key=lambda line: (
            line.center_y,
            line.left,
        ),
    )


def merge_boxes(
    boxes: Sequence[LayoutTextBox],
    separator: str = " ",
) -> LayoutTextBox:
    """
    Gộp nhiều OCR box thành một box duy nhất.
    """

    if not boxes:
        raise ValueError(
            "Không thể gộp danh sách box rỗng"
        )

    ordered = sort_left_to_right(
        boxes
    )

    text = separator.join(
        box.text.strip()
        for box in ordered
        if box.text.strip()
    ).strip()

    original_text = separator.join(
        box.original_text.strip()
        for box in ordered
        if box.original_text.strip()
    ).strip()

    left = min(
        box.left
        for box in ordered
    )

    top = min(
        box.top
        for box in ordered
    )

    right = max(
        box.right
        for box in ordered
    )

    bottom = max(
        box.bottom
        for box in ordered
    )

    total_area = sum(
        max(
            box.area,
            1.0,
        )
        for box in ordered
    )

    if total_area <= 0:
        confidence = 0.0
    else:
        confidence = sum(
            box.confidence
            * max(
                box.area,
                1.0,
            )
            for box in ordered
        ) / total_area

    return LayoutTextBox.from_rectangle(
        text=text,
        original_text=original_text,
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        confidence=confidence,
    )


def merge_same_line_boxes(
    boxes: Iterable[LayoutTextBox],
    minimum_vertical_overlap: float = 0.35,
    maximum_center_y_difference_ratio: float = 0.80,
) -> list[LayoutTextBox]:
    """
    Nhóm và gộp các box cùng dòng.
    """

    lines = group_boxes_into_lines(
        boxes=boxes,
        minimum_vertical_overlap=(
            minimum_vertical_overlap
        ),
        maximum_center_y_difference_ratio=(
            maximum_center_y_difference_ratio
        ),
    )

    return [
        line.to_layout_box()
        for line in lines
    ]


def find_boxes_in_vertical_range(
    boxes: Iterable[LayoutTextBox],
    minimum_y: float,
    maximum_y: float,
) -> list[LayoutTextBox]:
    """
    Lấy các box có tâm Y nằm trong khoảng chỉ định.
    """

    return [
        box
        for box in boxes
        if (
            minimum_y
            <= box.center_y
            <= maximum_y
        )
    ]


def find_boxes_in_horizontal_range(
    boxes: Iterable[LayoutTextBox],
    minimum_x: float,
    maximum_x: float,
) -> list[LayoutTextBox]:
    """
    Lấy các box có tâm X nằm trong khoảng chỉ định.
    """

    return [
        box
        for box in boxes
        if (
            minimum_x
            <= box.center_x
            <= maximum_x
        )
    ]


def find_boxes_inside_rectangle(
    boxes: Iterable[LayoutTextBox],
    left: float,
    top: float,
    right: float,
    bottom: float,
    minimum_overlap_ratio: float = 0.50,
) -> list[LayoutTextBox]:
    """
    Tìm box nằm trong một vùng hình chữ nhật.

    Box được chấp nhận nếu phần giao chiếm đủ tỷ lệ
    diện tích của chính box đó.
    """

    if right <= left:
        raise ValueError(
            "right phải lớn hơn left"
        )

    if bottom <= top:
        raise ValueError(
            "bottom phải lớn hơn top"
        )

    region = LayoutTextBox.from_rectangle(
        text="REGION",
        left=left,
        top=top,
        right=right,
        bottom=bottom,
    )

    results: list[LayoutTextBox] = []

    for box in boxes:
        intersection_left = max(
            region.left,
            box.left,
        )

        intersection_top = max(
            region.top,
            box.top,
        )

        intersection_right = min(
            region.right,
            box.right,
        )

        intersection_bottom = min(
            region.bottom,
            box.bottom,
        )

        intersection_width = max(
            0.0,
            intersection_right
            - intersection_left,
        )

        intersection_height = max(
            0.0,
            intersection_bottom
            - intersection_top,
        )

        intersection_area = (
            intersection_width
            * intersection_height
        )

        if box.area <= 0:
            continue

        overlap_ratio = (
            intersection_area
            / box.area
        )

        if (
            overlap_ratio
            >= minimum_overlap_ratio
        ):
            results.append(box)

    return sort_reading_order(
        results
    )


def prepare_layout_boxes(
    items: Iterable[Any],
    minimum_confidence: float = 0.05,
    remove_duplicates: bool = True,
    merge_lines: bool = False,
) -> list[LayoutTextBox]:
    """
    Chuẩn hóa đầu vào OCR trước khi parser sử dụng.

    Quy trình:
    1. Chuyển về LayoutTextBox.
    2. Lọc confidence thấp.
    3. Lọc box kích thước không hợp lệ.
    4. Loại box trùng.
    5. Có thể gộp các box cùng dòng.
    """

    boxes = convert_to_layout_boxes(
        items
    )

    boxes = filter_by_confidence(
        boxes,
        minimum_confidence=(
            minimum_confidence
        ),
    )

    boxes = filter_by_minimum_size(
        boxes
    )

    if remove_duplicates:
        boxes = remove_duplicate_boxes(
            boxes
        )

    if merge_lines:
        boxes = merge_same_line_boxes(
            boxes
        )

    return sort_reading_order(
        boxes
    )
