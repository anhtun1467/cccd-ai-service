from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


class GeometryRefiner:
    """Đo và sửa độ xiên nhỏ còn sót sau perspective transform.

    Perspective transform xử lý bốn góc của thẻ. Tuy nhiên, khi viền thẻ
    bị lóa, bo tròn hoặc nằm sát mép ảnh, bốn góc có thể lệch vài pixel và
    làm các dòng chữ vẫn dốc. Lớp này đo góc từ các nét ngang trong vùng
    chữ và dùng vertical shear để đưa dòng chữ về ngang mà không thay đổi
    tỷ lệ rộng/cao của ảnh thẻ.
    """

    MIN_CORRECTION_ANGLE = 0.45
    MAX_CORRECTION_ANGLE = 12.0
    MIN_RELIABLE_LINES = 4

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if image is None or not isinstance(image, np.ndarray):
            raise TypeError("Ảnh hiệu chỉnh phải là numpy.ndarray")
        if image.size == 0 or image.ndim not in (2, 3):
            raise ValueError("Ảnh hiệu chỉnh không có dữ liệu hợp lệ")

    @staticmethod
    def _normalize_horizontal_angle(angle: float) -> float:
        """Đưa hướng đường thẳng về miền (-90, 90] độ."""
        normalized = float(angle)
        while normalized <= -90.0:
            normalized += 180.0
        while normalized > 90.0:
            normalized -= 180.0
        return normalized

    @staticmethod
    def _weighted_median(
        values: list[float],
        weights: list[float],
    ) -> float:
        if not values or len(values) != len(weights):
            return 0.0

        order = np.argsort(np.asarray(values, dtype=np.float64))
        sorted_values = np.asarray(values, dtype=np.float64)[order]
        sorted_weights = np.asarray(weights, dtype=np.float64)[order]
        total_weight = float(np.sum(sorted_weights))
        if total_weight <= 0.0:
            return float(np.median(sorted_values))

        cumulative = np.cumsum(sorted_weights)
        index = int(np.searchsorted(cumulative, total_weight * 0.5))
        index = min(index, len(sorted_values) - 1)
        return float(sorted_values[index])

    def _summarize_angles(
        self,
        angles: list[float],
        weights: list[float],
        reference_width: float,
        source: str,
    ) -> dict[str, Any]:
        if not angles:
            return {
                "angleDegrees": 0.0,
                "confidence": 0.0,
                "lineCount": 0,
                "totalLineLength": 0.0,
                "medianAbsoluteDeviation": 0.0,
                "reliable": False,
                "source": source,
            }

        median = self._weighted_median(angles, weights)
        deviations = [abs(value - median) for value in angles]
        mad = self._weighted_median(deviations, weights)
        total_length = float(sum(weights))

        line_support = min(1.0, len(angles) / 12.0)
        length_support = min(
            1.0,
            total_length / max(reference_width * 2.25, 1.0),
        )
        dispersion_support = max(0.0, 1.0 - (mad / 3.5))
        confidence = (
            line_support * 0.35
            + length_support * 0.35
            + dispersion_support * 0.30
        )

        reliable = bool(
            len(angles) >= self.MIN_RELIABLE_LINES
            and total_length >= reference_width * 0.50
            and mad <= 4.0
            and confidence >= 0.42
        )

        return {
            "angleDegrees": round(median, 3),
            "confidence": round(float(confidence), 3),
            "lineCount": len(angles),
            "totalLineLength": round(total_length, 2),
            "medianAbsoluteDeviation": round(mad, 3),
            "reliable": reliable,
            "source": source,
        }

    def _estimate_from_hough(
        self,
        image: np.ndarray,
    ) -> dict[str, Any]:
        gray = (
            image.copy()
            if image.ndim == 2
            else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        )

        height, width = gray.shape[:2]
        analysis_width = min(width, 1000)
        if width > analysis_width:
            scale = analysis_width / float(width)
            gray = cv2.resize(
                gray,
                (analysis_width, max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )

        height, width = gray.shape[:2]
        x1 = int(round(width * 0.10))
        x2 = int(round(width * 0.96))
        y1 = int(round(height * 0.05))
        y2 = int(round(height * 0.95))
        region = gray[y1:y2, x1:x2]
        if region.size == 0:
            return self._summarize_angles([], [], float(width), "hough")

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )
        contrast = clahe.apply(region)
        blurred = cv2.GaussianBlur(contrast, (3, 3), 0)
        edges = cv2.Canny(
            blurred,
            threshold1=45,
            threshold2=145,
            apertureSize=3,
            L2gradient=True,
        )
        edges = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)),
        )

        region_height, region_width = region.shape[:2]
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 360.0,
            threshold=max(16, int(round(region_width * 0.022))),
            minLineLength=max(22, int(round(region_width * 0.032))),
            maxLineGap=max(5, int(round(region_width * 0.018))),
        )

        angles: list[float] = []
        weights: list[float] = []
        if lines is not None:
            for raw_line in lines.reshape(-1, 4):
                start_x, start_y, end_x, end_y = (
                    float(value) for value in raw_line
                )
                delta_x = end_x - start_x
                delta_y = end_y - start_y
                length = math.hypot(delta_x, delta_y)
                if length <= 0.0 or length > region_width * 0.72:
                    continue

                angle = self._normalize_horizontal_angle(
                    math.degrees(math.atan2(delta_y, delta_x))
                )
                if abs(angle) > self.MAX_CORRECTION_ANGLE:
                    continue

                # Loại các đoạn rất dài sát biên vì thường là viền thẻ,
                # không phải baseline hoặc nét ngang của chữ.
                near_horizontal_border = bool(
                    min(start_y, end_y) <= region_height * 0.025
                    or max(start_y, end_y) >= region_height * 0.975
                )
                if near_horizontal_border and length >= region_width * 0.45:
                    continue

                angles.append(angle)
                weights.append(length)

        return self._summarize_angles(
            angles,
            weights,
            float(region_width),
            "hough",
        )

    def _estimate_from_text_boxes(
        self,
        text_boxes: list[Any] | None,
        image_width: float,
    ) -> dict[str, Any]:
        angles: list[float] = []
        weights: list[float] = []

        for item in text_boxes or []:
            if isinstance(item, dict):
                box = item.get("box", [])
                confidence = float(item.get("confidence", 0.0) or 0.0)
            else:
                box = getattr(item, "box", [])
                confidence = float(getattr(item, "confidence", 0.0) or 0.0)

            try:
                points = np.asarray(box, dtype=np.float64).reshape(-1, 2)
            except (TypeError, ValueError):
                continue
            if points.shape[0] < 4:
                continue

            edge_pairs = ((points[0], points[1]), (points[3], points[2]))
            for start, end in edge_pairs:
                delta_x = float(end[0] - start[0])
                delta_y = float(end[1] - start[1])
                length = math.hypot(delta_x, delta_y)
                if length < max(12.0, image_width * 0.015):
                    continue
                angle = self._normalize_horizontal_angle(
                    math.degrees(math.atan2(delta_y, delta_x))
                )
                if abs(angle) > self.MAX_CORRECTION_ANGLE:
                    continue
                angles.append(angle)
                weights.append(length * max(0.35, confidence))

        return self._summarize_angles(
            angles,
            weights,
            max(float(image_width), 1.0),
            "ocr_boxes",
        )

    def estimate_text_skew(
        self,
        image: np.ndarray,
        text_boxes: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Ước lượng góc dốc của dòng chữ sau khi ảnh đã được làm phẳng."""
        self._validate_image(image)
        hough = self._estimate_from_hough(image)
        boxes = self._estimate_from_text_boxes(
            text_boxes,
            image_width=float(image.shape[1]),
        )

        selected = hough
        if not hough["reliable"] and boxes["reliable"]:
            selected = boxes
        elif hough["reliable"] and boxes["reliable"]:
            difference = abs(
                float(hough["angleDegrees"])
                - float(boxes["angleDegrees"])
            )
            boxes_were_axis_aligned = bool(
                abs(float(boxes["angleDegrees"])) < 0.15
                and abs(float(hough["angleDegrees"]))
                >= self.MIN_CORRECTION_ANGLE
            )
            # EasyOCR thường trả box ngang tuyệt đối cho góc rất nhỏ. Chỉ
            # kết hợp hai nguồn khi chúng thực sự đồng thuận.
            if difference <= 2.0 and not boxes_were_axis_aligned:
                hough_weight = max(float(hough["confidence"]), 0.01)
                box_weight = max(float(boxes["confidence"]), 0.01)
                combined_angle = (
                    float(hough["angleDegrees"]) * hough_weight
                    + float(boxes["angleDegrees"]) * box_weight
                ) / (hough_weight + box_weight)
                selected = {
                    **hough,
                    "angleDegrees": round(combined_angle, 3),
                    "confidence": round(
                        max(hough_weight, box_weight),
                        3,
                    ),
                    "source": "hough+ocr_boxes",
                }

        angle = float(selected.get("angleDegrees", 0.0))
        reliable = bool(
            selected.get("reliable")
            and self.MIN_CORRECTION_ANGLE
            <= abs(angle)
            <= self.MAX_CORRECTION_ANGLE
        )

        return {
            **selected,
            "reliable": reliable,
            "hough": hough,
            "ocrBoxes": boxes,
        }

    def build_correction_angles(
        self,
        estimate: dict[str, Any],
    ) -> list[float]:
        """Tạo tối đa hai mức sửa để OCR tự chọn mức tốt nhất."""
        if not estimate.get("reliable"):
            return []

        angle = float(estimate.get("angleDegrees", 0.0))
        if not (
            self.MIN_CORRECTION_ANGLE
            <= abs(angle)
            <= self.MAX_CORRECTION_ANGLE
        ):
            return []

        candidates = [round(angle, 3)]
        if abs(angle) >= 2.25:
            candidates.append(round(angle * 0.70, 3))
        return list(dict.fromkeys(candidates))

    def correct_vertical_shear(
        self,
        image: np.ndarray,
        correction_angle: float,
    ) -> np.ndarray:
        """Làm ngang baseline bằng affine shear, giữ nguyên kích thước."""
        self._validate_image(image)
        angle = float(correction_angle)
        if abs(angle) > self.MAX_CORRECTION_ANGLE:
            raise ValueError("Góc hiệu chỉnh vượt quá giới hạn an toàn")

        height, width = image.shape[:2]
        slope = math.tan(math.radians(angle))
        center_x = (width - 1) / 2.0
        matrix = np.array(
            [
                [1.0, 0.0, 0.0],
                [-slope, 1.0, slope * center_x],
            ],
            dtype=np.float32,
        )
        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )


geometry_refiner = GeometryRefiner()
