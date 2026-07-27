import cv2
import numpy as np


class PerspectiveTransformer:
    """
    Căn chỉnh phối cảnh CCCD từ 4 điểm góc.
    """

    def order_points(self, points: np.ndarray) -> np.ndarray:
        points = points.reshape(4, 2)

        rect = np.zeros((4, 2), dtype="float32")

        point_sum = points.sum(axis=1)
        rect[0] = points[np.argmin(point_sum)]  # top-left
        rect[2] = points[np.argmax(point_sum)]  # bottom-right

        point_diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(point_diff)]  # top-right
        rect[3] = points[np.argmax(point_diff)]  # bottom-left

        return rect

    def transform(self, image: np.ndarray, points: np.ndarray) -> np.ndarray:
        rect = self.order_points(points)

        top_left, top_right, bottom_right, bottom_left = rect

        width_a = np.linalg.norm(bottom_right - bottom_left)
        width_b = np.linalg.norm(top_right - top_left)
        max_width = int(max(width_a, width_b))

        height_a = np.linalg.norm(top_right - bottom_right)
        height_b = np.linalg.norm(top_left - bottom_left)
        max_height = int(max(height_a, height_b))

        destination = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype="float32",
        )

        matrix = cv2.getPerspectiveTransform(rect, destination)

        warped = cv2.warpPerspective(
            image,
            matrix,
            (max_width, max_height),
        )

        return warped
