from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class PerspectiveTransformer:
    """Làm phẳng và chuẩn hóa hình học mặt trước CCCD."""

    # Kích thước chuẩn ID-1: 85,60 x 53,98 mm.
    CARD_ASPECT_RATIO = 85.60 / 53.98
    MIN_OUTPUT_EDGE = 64

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
            "sourceCorners": rect.tolist(),
            "outputWidth": int(warped.shape[1]),
            "outputHeight": int(warped.shape[0]),
            "geometryRotationDegrees": geometry_rotation,
            "targetAspectRatio": round(self.CARD_ASPECT_RATIO, 6),
            "perspectiveMatrix": matrix.tolist(),
        }

        return warped, metadata

    def transform(self, image: np.ndarray, points: np.ndarray) -> np.ndarray:
        """API tương thích với mã cũ: chỉ trả ảnh đã làm phẳng."""
        warped, _ = self.transform_with_metadata(image, points)
        return warped
