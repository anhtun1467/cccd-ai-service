from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


class PerspectiveTransformer:
    """Làm phẳng và chuẩn hóa hình học mặt trước CCCD."""

    # Kích thước chuẩn ID-1: 85,60 x 53,98 mm.
    CARD_ASPECT_RATIO = 85.60 / 53.98
    MIN_OUTPUT_EDGE = 64

    def describe_source_geometry(
        self,
        image_shape: tuple[int, ...],
        points: np.ndarray,
    ) -> dict[str, Any]:
        """Đo mức phủ khung và độ méo của bốn góc nguồn."""
        image_height, image_width = image_shape[:2]
        rect = self.order_points(points)
        top_left, top_right, bottom_right, bottom_left = rect

        top_width = float(np.linalg.norm(top_right - top_left))
        bottom_width = float(np.linalg.norm(bottom_right - bottom_left))
        left_height = float(np.linalg.norm(bottom_left - top_left))
        right_height = float(np.linalg.norm(bottom_right - top_right))

        width_ratio = max(top_width, bottom_width) / max(
            min(top_width, bottom_width),
            1.0,
        )
        height_ratio = max(left_height, right_height) / max(
            min(left_height, right_height),
            1.0,
        )
        perspective_severity = max(
            abs(math.log(max(width_ratio, 1.0))),
            abs(math.log(max(height_ratio, 1.0))),
        )

        image_area = max(float(image_height * image_width), 1.0)
        source_area = abs(float(cv2.contourArea(rect)))
        x, y, box_width, box_height = cv2.boundingRect(rect)
        bounding_coverage = float(box_width * box_height) / image_area

        frame = np.array(
            [
                [0.0, 0.0],
                [float(image_width - 1), 0.0],
                [float(image_width - 1), float(image_height - 1)],
                [0.0, float(image_height - 1)],
            ],
            dtype=np.float32,
        )
        diagonal = max(math.hypot(image_width, image_height), 1.0)
        frame_deviation = float(
            np.mean(np.linalg.norm(rect - frame, axis=1)) / diagonal
        )

        mean_width = (top_width + bottom_width) / 2.0
        mean_height = (left_height + right_height) / 2.0
        short_edge = max(min(mean_width, mean_height), 1.0)
        measured_aspect = max(mean_width, mean_height) / short_edge

        edge_threshold = max(4.0, min(image_width, image_height) * 0.025)
        edge_touch_count = sum(
            bool(
                point[0] <= edge_threshold
                or point[0] >= image_width - 1 - edge_threshold
                or point[1] <= edge_threshold
                or point[1] >= image_height - 1 - edge_threshold
            )
            for point in rect
        )

        return {
            "sourceCoverageRatio": round(source_area / image_area, 4),
            "sourceBoundingCoverageRatio": round(bounding_coverage, 4),
            "sourceMeasuredAspectRatio": round(measured_aspect, 4),
            "oppositeWidthRatio": round(width_ratio, 4),
            "oppositeHeightRatio": round(height_ratio, 4),
            "perspectiveSeverity": round(perspective_severity, 4),
            "frameCornerDeviation": round(frame_deviation, 4),
            "edgeTouchCount": int(edge_touch_count),
            "sourceBoundingBox": [
                int(x),
                int(y),
                int(box_width),
                int(box_height),
            ],
        }

    def order_points(self, points: np.ndarray) -> np.ndarray:
        """Sắp xếp bốn góc theo TL, TR, BR, BL."""
        array = np.asarray(points, dtype="float32").reshape(-1, 2)

        if array.shape != (4, 2):
            raise ValueError("Cần đúng 4 điểm góc để hiệu chỉnh phối cảnh")

        x_sorted = array[np.argsort(array[:, 0])]
        left_most = x_sorted[:2]
        right_most = x_sorted[2:]

        left_most = left_most[np.argsort(left_most[:, 1])]
        top_left, bottom_left = left_most

        distances = np.linalg.norm(right_most - top_left, axis=1)
        bottom_right = right_most[np.argmax(distances)]
        top_right = right_most[np.argmin(distances)]

        rect = np.array(
            [top_left, top_right, bottom_right, bottom_left],
            dtype="float32",
        )

        return rect

    def _calculate_output_size(
        self,
        rect: np.ndarray,
    ) -> tuple[int, int]:
        top_left, top_right, bottom_right, bottom_left = rect

        measured_width = max(
            np.linalg.norm(bottom_right - bottom_left),
            np.linalg.norm(top_right - top_left),
        )
        measured_height = max(
            np.linalg.norm(top_right - bottom_right),
            np.linalg.norm(top_left - bottom_left),
        )

        long_edge = max(measured_width, measured_height)
        if long_edge < self.MIN_OUTPUT_EDGE:
            raise ValueError("Vùng CCCD quá nhỏ để hiệu chỉnh phối cảnh")

        # Ép về đúng tỷ lệ vật lý của CCCD để loại bỏ co kéo do góc chụp.
        short_edge = max(
            self.MIN_OUTPUT_EDGE,
            long_edge / self.CARD_ASPECT_RATIO,
        )

        if measured_width >= measured_height:
            return int(round(long_edge)), int(round(short_edge))

        return int(round(short_edge)), int(round(long_edge))

    def transform_with_metadata(
        self,
        image: np.ndarray,
        points: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Biến tứ giác CCCD thành hình chữ nhật và xoay về chiều ngang.

        Việc xoay 180 độ theo nội dung được thực hiện ở OCR pipeline, nơi
        có thể dùng chính các nhãn CCCD để chọn chiều có độ tin cậy cao hơn.
        """
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Ảnh đầu vào không hợp lệ")

        rect = self.order_points(points)
        output_width, output_height = self._calculate_output_size(rect)

        destination = np.array(
            [
                [0, 0],
                [output_width - 1, 0],
                [output_width - 1, output_height - 1],
                [0, output_height - 1],
            ],
            dtype="float32",
        )

        matrix = cv2.getPerspectiveTransform(rect, destination)
        warped = cv2.warpPerspective(
            image,
            matrix,
            (output_width, output_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        geometry_rotation = 0
        if warped.shape[0] > warped.shape[1]:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
            geometry_rotation = 90

        metadata: dict[str, Any] = {
            "perspectiveApplied": True,
            "candidateName": "perspective_contour",
            "sourceCorners": rect.tolist(),
            "outputWidth": int(warped.shape[1]),
            "outputHeight": int(warped.shape[0]),
            "geometryRotationDegrees": geometry_rotation,
            "targetAspectRatio": round(self.CARD_ASPECT_RATIO, 6),
            "perspectiveMatrix": matrix.tolist(),
            **self.describe_source_geometry(image.shape, rect),
        }

        return warped, metadata

    def normalize_full_frame_with_metadata(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Giữ toàn bộ khung ảnh khi đầu vào vốn đã là một ảnh thẻ.

        Đây là ứng viên an toàn cho ảnh có thẻ chạm sát bốn mép. Trong tình
        huống đó contour dễ bám nhầm vào phần bo góc và perspective transform
        có thể cắt chữ, trong khi resize toàn khung vẫn giữ đủ nội dung.
        """
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Ảnh đầu vào không hợp lệ")

        source_height, source_width = image.shape[:2]
        normalized = image.copy()
        geometry_rotation = 0
        if source_height > source_width:
            normalized = cv2.rotate(normalized, cv2.ROTATE_90_CLOCKWISE)
            geometry_rotation = 90

        height, width = normalized.shape[:2]
        target_width = max(
            width,
            int(round(self.MIN_OUTPUT_EDGE * self.CARD_ASPECT_RATIO)),
        )
        target_height = max(
            self.MIN_OUTPUT_EDGE,
            int(round(target_width / self.CARD_ASPECT_RATIO)),
        )
        interpolation = (
            cv2.INTER_CUBIC
            if target_width > width or target_height > height
            else cv2.INTER_AREA
        )
        normalized = cv2.resize(
            normalized,
            (target_width, target_height),
            interpolation=interpolation,
        )

        source_corners = np.array(
            [
                [0.0, 0.0],
                [float(source_width - 1), 0.0],
                [float(source_width - 1), float(source_height - 1)],
                [0.0, float(source_height - 1)],
            ],
            dtype=np.float32,
        )
        metadata: dict[str, Any] = {
            "perspectiveApplied": False,
            "frameNormalizationApplied": True,
            "candidateName": "full_frame",
            "sourceCorners": source_corners.tolist(),
            "outputWidth": int(normalized.shape[1]),
            "outputHeight": int(normalized.shape[0]),
            "geometryRotationDegrees": geometry_rotation,
            "targetAspectRatio": round(self.CARD_ASPECT_RATIO, 6),
            "perspectiveMatrix": None,
            "resizeScaleX": round(target_width / float(width), 6),
            "resizeScaleY": round(target_height / float(height), 6),
            **self.describe_source_geometry(image.shape, source_corners),
        }

        return normalized, metadata

    def transform(self, image: np.ndarray, points: np.ndarray) -> np.ndarray:
        """API tương thích với mã cũ: chỉ trả ảnh đã làm phẳng."""
        warped, _ = self.transform_with_metadata(image, points)
        return warped
