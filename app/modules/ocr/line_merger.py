from __future__ import annotations

from typing import Any

from app.modules.ocr.text_normalizer import OCRTextNormalizer


class OCRLineMerger:
    """
    Ghép các text box OCR thuộc cùng một dòng dựa trên tọa độ.
    """

    def __init__(
        self,
        vertical_tolerance_ratio: float = 0.6,
        maximum_horizontal_gap_ratio: float = 3.5,
    ) -> None:
        self.vertical_tolerance_ratio = vertical_tolerance_ratio
        self.maximum_horizontal_gap_ratio = maximum_horizontal_gap_ratio

    @staticmethod
    def _bounds(box: list[list[float]]) -> tuple[float, float, float, float]:
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]

        return min(xs), min(ys), max(xs), max(ys)

    def merge(self, text_boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        valid_boxes: list[dict[str, Any]] = []

        for item in text_boxes:
            box = item.get("box")
            text = OCRTextNormalizer.normalize(item.get("text", ""))

            if not box or len(box) < 4 or not text:
                continue

            left, top, right, bottom = self._bounds(box)
            height = max(bottom - top, 1.0)

            valid_boxes.append({
                **item,
                "text": text,
                "_left": left,
                "_top": top,
                "_right": right,
                "_bottom": bottom,
                "_height": height,
                "_center_y": (top + bottom) / 2,
            })

        valid_boxes.sort(
            key=lambda item: (
                item["_center_y"],
                item["_left"],
            )
        )

        rows: list[list[dict[str, Any]]] = []

        for current in valid_boxes:
            selected_row: list[dict[str, Any]] | None = None

            for row in rows:
                row_center_y = sum(
                    item["_center_y"] for item in row
                ) / len(row)

                row_height = sum(
                    item["_height"] for item in row
                ) / len(row)

                tolerance = max(
                    row_height,
                    current["_height"],
                ) * self.vertical_tolerance_ratio

                if abs(current["_center_y"] - row_center_y) <= tolerance:
                    selected_row = row
                    break

            if selected_row is None:
                rows.append([current])
            else:
                selected_row.append(current)

        merged_lines: list[dict[str, Any]] = []

        for row in rows:
            row.sort(key=lambda item: item["_left"])

            segments: list[list[dict[str, Any]]] = []
            current_segment: list[dict[str, Any]] = []

            for item in row:
                if not current_segment:
                    current_segment.append(item)
                    continue

                previous = current_segment[-1]
                gap = item["_left"] - previous["_right"]

                average_height = (
                    item["_height"] + previous["_height"]
                ) / 2

                maximum_gap = (
                    average_height
                    * self.maximum_horizontal_gap_ratio
                )

                if gap <= maximum_gap:
                    current_segment.append(item)
                else:
                    segments.append(current_segment)
                    current_segment = [item]

            if current_segment:
                segments.append(current_segment)

            for segment in segments:
                merged_text = OCRTextNormalizer.normalize(
                    " ".join(item["text"] for item in segment)
                )

                confidence_values = [
                    float(item.get("confidence", 0.0))
                    for item in segment
                ]

                confidence = (
                    sum(confidence_values) / len(confidence_values)
                    if confidence_values
                    else 0.0
                )

                merged_lines.append({
                    "text": merged_text,
                    "confidence": round(confidence, 4),
                    "box": [
                        [
                            min(item["_left"] for item in segment),
                            min(item["_top"] for item in segment),
                        ],
                        [
                            max(item["_right"] for item in segment),
                            min(item["_top"] for item in segment),
                        ],
                        [
                            max(item["_right"] for item in segment),
                            max(item["_bottom"] for item in segment),
                        ],
                        [
                            min(item["_left"] for item in segment),
                            max(item["_bottom"] for item in segment),
                        ],
                    ],
                    "sourceCount": len(segment),
                })

        merged_lines.sort(
            key=lambda item: (
                item["box"][0][1],
                item["box"][0][0],
            )
        )

        return merged_lines