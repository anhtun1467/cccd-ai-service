from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence


Point = tuple[float, float]
Quadrilateral = tuple[Point, Point, Point, Point]


@dataclass(frozen=True, slots=True)
class LayoutTextBox:
    """
    Đại diện cho một text box OCR trong hệ tọa độ ảnh.

    Thuộc tính:
        text:
            Nội dung văn bản đã được chuẩn hóa.

        original_text:
            Nội dung OCR ban đầu trước khi chuẩn hóa.

        confidence:
            Độ tin cậy OCR trong khoảng 0.0 đến 1.0.

        box:
            Bốn điểm của vùng OCR theo thứ tự:
            top-left, top-right, bottom-right, bottom-left.
    """

    text: str
    original_text: str
    confidence: float
    box: Quadrilateral

    @property
    def left(self) -> float:
        """Tọa độ trái nhất của box."""

        return min(point[0] for point in self.box)

    @property
    def right(self) -> float:
        """Tọa độ phải nhất của box."""

        return max(point[0] for point in self.box)

    @property
    def top(self) -> float:
        """Tọa độ trên cùng của box."""

        return min(point[1] for point in self.box)

    @property
    def bottom(self) -> float:
        """Tọa độ dưới cùng của box."""

        return max(point[1] for point in self.box)

    @property
    def width(self) -> float:
        """Chiều rộng bao ngoài của box."""

        return max(0.0, self.right - self.left)

    @property
    def height(self) -> float:
        """Chiều cao bao ngoài của box."""

        return max(0.0, self.bottom - self.top)

    @property
    def area(self) -> float:
        """Diện tích hình chữ nhật bao ngoài."""

        return self.width * self.height

    @property
    def center_x(self) -> float:
        """Tọa độ tâm theo trục X."""

        return (self.left + self.right) / 2.0

    @property
    def center_y(self) -> float:
        """Tọa độ tâm theo trục Y."""

        return (self.top + self.bottom) / 2.0

    @property
    def center(self) -> Point:
        """Tọa độ tâm của box."""

        return self.center_x, self.center_y

    @property
    def is_empty(self) -> bool:
        """Kiểm tra box có nội dung sử dụng được hay không."""

        return (
            not self.text.strip()
            or self.width <= 0
            or self.height <= 0
        )

    def to_rectangle(self) -> tuple[float, float, float, float]:
        """
        Trả về hình chữ nhật bao ngoài theo dạng:

        left, top, right, bottom
        """

        return self.left, self.top, self.right, self.bottom

    def to_dict(self) -> dict[str, Any]:
        """Chuyển đối tượng về dictionary JSON-safe."""

        return {
            "text": self.text,
            "originalText": self.original_text,
            "confidence": round(self.confidence, 4),
            "box": [
                [float(x), float(y)]
                for x, y in self.box
            ],
            "left": float(self.left),
            "top": float(self.top),
            "right": float(self.right),
            "bottom": float(self.bottom),
            "width": float(self.width),
            "height": float(self.height),
            "center": [
                float(self.center_x),
                float(self.center_y),
            ],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> LayoutTextBox:
        """
        Tạo LayoutTextBox từ dictionary.

        Hỗ trợ dữ liệu dạng:

        {
            "text": "...",
            "originalText": "...",
            "confidence": 0.95,
            "box": [[x, y], ...]
        }
        """

        if not isinstance(data, dict):
            raise TypeError(
                "Dữ liệu text box phải là dictionary"
            )

        text = str(
            data.get("text", "")
        ).strip()

        original_text = str(
            data.get(
                "originalText",
                data.get(
                    "original_text",
                    text,
                ),
            )
        ).strip()

        confidence = cls.normalize_confidence(
            data.get("confidence", 0.0)
        )

        box = cls.normalize_box(
            data.get("box")
        )

        return cls(
            text=text,
            original_text=original_text,
            confidence=confidence,
            box=box,
        )

    @classmethod
    def from_object(
        cls,
        item: Any,
    ) -> LayoutTextBox:
        """
        Tạo LayoutTextBox từ object OCR hoặc dictionary.

        Hàm này giúp tương thích với:
        - OCRTextBox dataclass hiện tại.
        - Dictionary JSON.
        - Object có thuộc tính text, confidence và box.
        """

        if isinstance(item, cls):
            return item

        if isinstance(item, dict):
            return cls.from_dict(item)

        text = str(
            getattr(item, "text", "")
        ).strip()

        original_text = str(
            getattr(
                item,
                "original_text",
                getattr(
                    item,
                    "originalText",
                    text,
                ),
            )
        ).strip()

        confidence = cls.normalize_confidence(
            getattr(
                item,
                "confidence",
                0.0,
            )
        )

        box = cls.normalize_box(
            getattr(item, "box", None)
        )

        return cls(
            text=text,
            original_text=original_text,
            confidence=confidence,
            box=box,
        )

    @classmethod
    def from_rectangle(
        cls,
        text: str,
        left: float,
        top: float,
        right: float,
        bottom: float,
        confidence: float = 0.0,
        original_text: str | None = None,
    ) -> LayoutTextBox:
        """
        Tạo box từ hình chữ nhật left, top, right, bottom.
        """

        if right <= left:
            raise ValueError(
                "right phải lớn hơn left"
            )

        if bottom <= top:
            raise ValueError(
                "bottom phải lớn hơn top"
            )

        box: Quadrilateral = (
            (float(left), float(top)),
            (float(right), float(top)),
            (float(right), float(bottom)),
            (float(left), float(bottom)),
        )

        return cls(
            text=text.strip(),
            original_text=(
                original_text.strip()
                if original_text is not None
                else text.strip()
            ),
            confidence=cls.normalize_confidence(
                confidence
            ),
            box=box,
        )

    @staticmethod
    def normalize_confidence(
        confidence: Any,
    ) -> float:
        """
        Chuẩn hóa confidence về khoảng 0.0 đến 1.0.
        """

        try:
            value = float(confidence)
        except (TypeError, ValueError):
            return 0.0

        if value < 0.0:
            return 0.0

        if value > 1.0:
            return 1.0

        return value

    @staticmethod
    def normalize_box(
        raw_box: Any,
    ) -> Quadrilateral:
        """
        Chuẩn hóa box về đúng bốn điểm.

        Hỗ trợ:
        - [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        - tuple tương đương
        - object NumPy có phương thức tolist()
        """

        if raw_box is None:
            raise ValueError(
                "Text box không có tọa độ"
            )

        if hasattr(raw_box, "tolist"):
            raw_box = raw_box.tolist()

        if not isinstance(
            raw_box,
            (list, tuple),
        ):
            raise TypeError(
                "Box phải là list hoặc tuple"
            )

        if len(raw_box) != 4:
            raise ValueError(
                "Box phải có đúng 4 điểm"
            )

        points: list[Point] = []

        for index, point in enumerate(raw_box):
            if hasattr(point, "tolist"):
                point = point.tolist()

            if not isinstance(
                point,
                (list, tuple),
            ):
                raise TypeError(
                    f"Điểm thứ {index} không hợp lệ"
                )

            if len(point) < 2:
                raise ValueError(
                    f"Điểm thứ {index} phải có x và y"
                )

            try:
                x = float(point[0])
                y = float(point[1])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Tọa độ điểm thứ {index} không hợp lệ"
                ) from error

            points.append((x, y))

        normalized = LayoutTextBox.order_points(
            points
        )

        return (
            normalized[0],
            normalized[1],
            normalized[2],
            normalized[3],
        )

    @staticmethod
    def order_points(
        points: Sequence[Point],
    ) -> list[Point]:
        """
        Sắp xếp bốn điểm theo thứ tự:

        top-left
        top-right
        bottom-right
        bottom-left
        """

        if len(points) != 4:
            raise ValueError(
                "Cần đúng 4 điểm để sắp xếp"
            )

        sorted_by_y = sorted(
            points,
            key=lambda point: (
                point[1],
                point[0],
            ),
        )

        top_points = sorted(
            sorted_by_y[:2],
            key=lambda point: point[0],
        )

        bottom_points = sorted(
            sorted_by_y[2:],
            key=lambda point: point[0],
        )

        top_left = top_points[0]
        top_right = top_points[1]
        bottom_left = bottom_points[0]
        bottom_right = bottom_points[1]

        return [
            top_left,
            top_right,
            bottom_right,
            bottom_left,
        ]


def convert_to_layout_boxes(
    items: Iterable[Any],
    remove_empty: bool = True,
) -> list[LayoutTextBox]:
    """
    Chuyển danh sách OCR box sang LayoutTextBox.

    Các phần tử lỗi sẽ bị bỏ qua để không làm hỏng toàn bộ pipeline.
    """

    results: list[LayoutTextBox] = []

    for item in items:
        try:
            text_box = LayoutTextBox.from_object(
                item
            )
        except (
            TypeError,
            ValueError,
            AttributeError,
        ):
            continue

        if remove_empty and text_box.is_empty:
            continue

        results.append(text_box)

    return results