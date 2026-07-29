from __future__ import annotations

import re
from typing import Any

from app.modules.ocr.text_normalizer import OCRTextNormalizer


class OCRLineMerger:
    """
    Ghép OCR text box theo cùng dòng nhưng tránh ghép nhầm
    các trường độc lập trên CCCD.

    Điểm khác bản cũ:
    - Dùng vertical overlap thay vì chỉ dựa vào center_y.
    - Không ghép khi hai box chỉ chạm nhau ở mép dòng.
    - Giới hạn horizontal gap chặt hơn.
    - Tách segment khi box sau bắt đầu một nhãn trường mới.
    - Không ghép vùng ngày hết hạn bên trái với địa chỉ bên phải.
    """

    FIELD_LABEL_PATTERN = re.compile(
        r"^(?:"
        r"so(?:\s*/\s*no)?|"
        r"ho\s*va\s*ten|full\s*name|"
        r"ngay\s*sinh|date\s*of\s*birth|"
        r"gioi\s*tinh|sex|"
        r"quoc\s*tich|nationality|"
        r"que\s*quan|place\s*of\s*origin|"
        r"noi\s*thuong\s*tru|place\s*of\s*residence|"
        r"co\s*gia\s*tr[iyj1l]\s*den|date\s*of\s*expiry"
        r")\b",
        flags=re.IGNORECASE,
    )

    def __init__(
        self,
        vertical_tolerance_ratio: float = 0.35,
        maximum_horizontal_gap_ratio: float = 1.8,
        minimum_vertical_overlap_ratio: float = 0.45,
    ) -> None:
        self.vertical_tolerance_ratio = float(
            vertical_tolerance_ratio
        )
        self.maximum_horizontal_gap_ratio = float(
            maximum_horizontal_gap_ratio
        )
        self.minimum_vertical_overlap_ratio = float(
            minimum_vertical_overlap_ratio
        )

    @staticmethod
    def _bounds(
        box: list[list[float]],
    ) -> tuple[float, float, float, float]:
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]

        return (
            min(xs),
            min(ys),
            max(xs),
            max(ys),
        )

    @staticmethod
    def _vertical_overlap_ratio(
        first: dict[str, Any],
        second: dict[str, Any],
    ) -> float:
        overlap = max(
            0.0,
            min(first["_bottom"], second["_bottom"])
            - max(first["_top"], second["_top"]),
        )

        smaller_height = max(
            min(first["_height"], second["_height"]),
            1.0,
        )

        return overlap / smaller_height

    def _same_row(
        self,
        current: dict[str, Any],
        row: list[dict[str, Any]],
    ) -> bool:
        row_top = min(item["_top"] for item in row)
        row_bottom = max(item["_bottom"] for item in row)
        row_height = max(row_bottom - row_top, 1.0)
        row_center_y = (row_top + row_bottom) / 2.0

        pseudo_row = {
            "_top": row_top,
            "_bottom": row_bottom,
            "_height": row_height,
        }

        overlap_ratio = self._vertical_overlap_ratio(
            current,
            pseudo_row,
        )

        center_tolerance = max(
            row_height,
            current["_height"],
        ) * self.vertical_tolerance_ratio

        center_close = (
            abs(current["_center_y"] - row_center_y)
            <= center_tolerance
        )

        return (
            overlap_ratio
            >= self.minimum_vertical_overlap_ratio
            and center_close
        )

    @classmethod
    def _starts_new_field(
        cls,
        text: str,
    ) -> bool:
        normalized = OCRTextNormalizer.normalize(
            text
        )
        return bool(
            normalized
            and cls.FIELD_LABEL_PATTERN.search(
                normalized
            )
        )

    @staticmethod
    def _is_left_expiry_to_right_address_pair(
        previous: dict[str, Any],
        current: dict[str, Any],
    ) -> bool:
        """
        Trên CCCD, ngày hết hạn nằm bên trái còn địa chỉ nằm bên phải.
        Hai vùng có thể gần cùng y nhưng không thuộc cùng một dòng logic.
        """

        previous_left = previous["_left"]
        current_left = current["_left"]

        previous_text = str(
            previous.get("text", "")
        ).lower()
        current_text = str(
            current.get("text", "")
        ).lower()

        previous_looks_expiry = bool(
            re.search(
                r"(?:co|gia|tri|den|expiry|\d{1,2}/\d{1,2}/\d{4})",
                previous_text,
            )
        )

        current_looks_address = bool(
            re.search(
                r"(?:noi\s*thuong\s*tru|place\s*of\s*residence|"
                r"thi\s*tran|phuong|xa|khom|thon|tam\s*binh)",
                current_text,
            )
        )

        return (
            previous_left < 330
            and current_left >= 300
            and previous_looks_expiry
            and current_looks_address
        )

    def _can_merge_horizontally(
        self,
        previous: dict[str, Any],
        current: dict[str, Any],
    ) -> bool:
        if self._is_left_expiry_to_right_address_pair(
            previous,
            current,
        ):
            return False

        gap = current["_left"] - previous["_right"]

        # Box chồng nhau hoặc kề nhau.
        if gap <= 0:
            return True

        average_height = (
            previous["_height"] + current["_height"]
        ) / 2.0

        maximum_gap = (
            average_height
            * self.maximum_horizontal_gap_ratio
        )

        if gap > maximum_gap:
            return False

        # Không nối một nhãn trường mới vào segment đã có nội dung,
        # trừ các nhãn song ngữ nằm thực sự sát nhau.
        if self._starts_new_field(current["text"]):
            previous_text = str(
                previous.get("text", "")
            )

            bilingual_pair = bool(
                re.search(
                    r"(?:gioi\s*tinh|quoc\s*tich|que\s*quan|"
                    r"noi\s*thuong\s*tru|ngay\s*sinh|ho\s*va\s*ten)",
                    previous_text,
                    flags=re.IGNORECASE,
                )
            )

            if not bilingual_pair:
                return False

        return True

    def merge(
        self,
        text_boxes: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        valid_boxes: list[dict[str, Any]] = []

        for item in text_boxes:
            box = item.get("box")
            text = OCRTextNormalizer.normalize(
                item.get("text", "")
            )

            if not box or len(box) < 4 or not text:
                continue

            try:
                left, top, right, bottom = (
                    self._bounds(box)
                )
            except (TypeError, ValueError, IndexError):
                continue

            width = max(right - left, 1.0)
            height = max(bottom - top, 1.0)

            valid_boxes.append(
                {
                    **item,
                    "text": text,
                    "_left": left,
                    "_top": top,
                    "_right": right,
                    "_bottom": bottom,
                    "_width": width,
                    "_height": height,
                    "_center_y": (top + bottom) / 2.0,
                }
            )

        valid_boxes.sort(
            key=lambda item: (
                item["_top"],
                item["_left"],
            )
        )

        rows: list[list[dict[str, Any]]] = []

        for current in valid_boxes:
            candidate_rows = [
                row
                for row in rows
                if self._same_row(current, row)
            ]

            if not candidate_rows:
                rows.append([current])
                continue

            selected_row = min(
                candidate_rows,
                key=lambda row: abs(
                    current["_center_y"]
                    - (
                        sum(
                            item["_center_y"]
                            for item in row
                        )
                        / len(row)
                    )
                ),
            )
            selected_row.append(current)

        merged_lines: list[dict[str, Any]] = []

        for row in rows:
            row.sort(key=lambda item: item["_left"])

            segments: list[list[dict[str, Any]]] = []
            current_segment: list[
                dict[str, Any]
            ] = []

            for item in row:
                if not current_segment:
                    current_segment = [item]
                    continue

                previous = current_segment[-1]

                if self._can_merge_horizontally(
                    previous,
                    item,
                ):
                    current_segment.append(item)
                else:
                    segments.append(current_segment)
                    current_segment = [item]

            if current_segment:
                segments.append(current_segment)

            for segment in segments:
                merged_text = (
                    OCRTextNormalizer.normalize(
                        " ".join(
                            item["text"]
                            for item in segment
                        )
                    )
                )

                if not merged_text:
                    continue

                confidence_values: list[float] = []

                for item in segment:
                    try:
                        confidence_values.append(
                            float(
                                item.get(
                                    "confidence",
                                    0.0,
                                )
                            )
                        )
                    except (TypeError, ValueError):
                        continue

                confidence = (
                    sum(confidence_values)
                    / len(confidence_values)
                    if confidence_values
                    else 0.0
                )

                merged_lines.append(
                    {
                        "text": merged_text,
                        "confidence": round(
                            confidence,
                            4,
                        ),
                        "box": [
                            [
                                min(
                                    item["_left"]
                                    for item in segment
                                ),
                                min(
                                    item["_top"]
                                    for item in segment
                                ),
                            ],
                            [
                                max(
                                    item["_right"]
                                    for item in segment
                                ),
                                min(
                                    item["_top"]
                                    for item in segment
                                ),
                            ],
                            [
                                max(
                                    item["_right"]
                                    for item in segment
                                ),
                                max(
                                    item["_bottom"]
                                    for item in segment
                                ),
                            ],
                            [
                                min(
                                    item["_left"]
                                    for item in segment
                                ),
                                max(
                                    item["_bottom"]
                                    for item in segment
                                ),
                            ],
                        ],
                        "sourceCount": len(segment),
                    }
                )

        merged_lines.sort(
            key=lambda item: (
                item["box"][0][1],
                item["box"][0][0],
            )
        )

        return merged_lines