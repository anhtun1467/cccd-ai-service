from __future__ import annotations

import cv2
import numpy as np


class ContourDetector:
    """
    Tìm contour CCCD bằng chiến lược kết hợp:
    - Threshold vùng sáng
    - Morphology Close để nối viền ngoài thẻ
    - Lọc contour theo diện tích và tỉ lệ CCCD
    """

    def __init__(
        self,
        max_contours: int = 80,
        epsilon_factor: float = 0.03,
        min_area_ratio: float = 0.20,
        min_aspect_ratio: float = 1.35,
        max_aspect_ratio: float = 2.35,
        padding: int = 55,
    ):
        self.max_contours = max_contours
        self.epsilon_factor = epsilon_factor
        self.min_area_ratio = min_area_ratio
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.padding = padding

    def build_card_mask(self, resized_image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)

        _, thresh = cv2.threshold(
            blur,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))

        closed = cv2.morphologyEx(
            thresh,
            cv2.MORPH_CLOSE,
            kernel,
        )

        opened = cv2.morphologyEx(
            closed,
            cv2.MORPH_OPEN,
            kernel,
        )

        return opened

    def find_contours(self, mask_image: np.ndarray) -> list[np.ndarray]:
        contours, _ = cv2.findContours(
            mask_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        contours = sorted(
            contours,
            key=cv2.contourArea,
            reverse=True,
        )

        return contours[: self.max_contours]

    def is_valid_card_contour(
        self,
        contour: np.ndarray,
        image_shape: tuple[int, ...],
    ) -> bool:
        image_height, image_width = image_shape[:2]
        image_area = image_height * image_width

        area = cv2.contourArea(contour)

        if area < image_area * self.min_area_ratio:
            return False

        (_, _), (rect_width, rect_height), _ = cv2.minAreaRect(contour)

        short_edge = min(rect_width, rect_height)
        long_edge = max(rect_width, rect_height)

        if short_edge <= 0:
            return False

        # minAreaRect giữ đúng tỷ lệ ngay cả khi thẻ xoay 90 độ hoặc
        # chụp chéo; boundingRect cũ làm thẻ nghiêng bị sai tỷ lệ.
        aspect_ratio = long_edge / float(short_edge)

        if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
            return False

        return True

    def approximate_contour(self, contour: np.ndarray) -> np.ndarray:
        perimeter = cv2.arcLength(contour, True)

        return cv2.approxPolyDP(
            contour,
            self.epsilon_factor * perimeter,
            True,
        )

    def find_quadrilateral(
        self,
        contour: np.ndarray,
    ) -> np.ndarray | None:
        """Tìm bốn góc thật của thẻ bằng nhiều mức xấp xỉ."""
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)

        factors = (
            self.epsilon_factor,
            0.01,
            0.015,
            0.02,
            0.025,
            0.035,
            0.04,
            0.05,
        )

        for factor in dict.fromkeys(factors):
            approx = cv2.approxPolyDP(
                hull,
                factor * perimeter,
                True,
            )
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue

            if cv2.contourArea(approx) < cv2.contourArea(contour) * 0.75:
                continue

            return approx.astype(np.float32)

        return None

    def expand_quadrilateral(
        self,
        points: np.ndarray,
        image_shape: tuple[int, ...],
    ) -> np.ndarray:
        """Nới rất nhẹ bốn góc để không cắt dấu ở sát mép thẻ."""
        quad = np.asarray(points, dtype=np.float32).reshape(4, 2)
        center = quad.mean(axis=0)

        side_lengths = [
            np.linalg.norm(quad[(index + 1) % 4] - quad[index])
            for index in range(4)
        ]
        short_edge = max(min(side_lengths), 1.0)

        # Padding 55 px trước đây đưa cả nền vào ảnh và làm mất hiệu lực
        # của perspective transform. Chỉ nới tối đa 1,5% cạnh ngắn.
        effective_padding = min(float(self.padding), short_edge * 0.015)
        scale = 1.0 + (2.0 * effective_padding / short_edge)
        expanded = center + (quad - center) * scale

        image_height, image_width = image_shape[:2]
        expanded[:, 0] = np.clip(expanded[:, 0], 0, image_width - 1)
        expanded[:, 1] = np.clip(expanded[:, 1], 0, image_height - 1)

        return expanded.reshape(4, 1, 2).astype(np.float32)

    def contour_to_quadrilateral(
        self,
        contour: np.ndarray,
        image_shape: tuple[int, ...],
    ) -> np.ndarray:
        """Trả bốn góc phối cảnh; fallback là rotated rectangle."""
        quadrilateral = self.find_quadrilateral(contour)

        if quadrilateral is None:
            rotated_rect = cv2.minAreaRect(contour)
            quadrilateral = cv2.boxPoints(rotated_rect).reshape(4, 1, 2)

        return self.expand_quadrilateral(
            quadrilateral,
            image_shape,
        )

    def add_padding_to_box(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        image_shape: tuple[int, ...],
    ) -> np.ndarray:
        image_height, image_width = image_shape[:2]

        x1 = max(x - self.padding, 0)
        y1 = max(y - self.padding, 0)
        x2 = min(x + w + self.padding, image_width - 1)
        y2 = min(y + h + self.padding, image_height - 1)

        return np.array(
            [
                [[x1, y1]],
                [[x2, y1]],
                [[x2, y2]],
                [[x1, y2]],
            ],
            dtype=np.int32,
        )

    def find_card_contour_from_image(
        self,
        resized_image: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray, list[np.ndarray]]:
        mask = self.build_card_mask(resized_image)
        contours = self.find_contours(mask)

        for contour in contours:
            if not self.is_valid_card_contour(contour, resized_image.shape):
                continue

            # Dùng bốn góc thật để làm phẳng phối cảnh. Đây là điểm khác
            # biệt quan trọng so với bounding box ngang của bản cũ.
            return (
                self.contour_to_quadrilateral(
                    contour,
                    resized_image.shape,
                ),
                mask,
                contours,
            )

        return None, mask, contours
