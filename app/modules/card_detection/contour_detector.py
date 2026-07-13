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
        padding: int = 30,
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

        x, y, w, h = cv2.boundingRect(contour)

        if h == 0:
            return False

        aspect_ratio = w / float(h)

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

            approx = self.approximate_contour(contour)

            x, y, w, h = cv2.boundingRect(contour)

            # Nếu approx ra 4 điểm, vẫn dùng bounding box có padding
            # để tránh bị cắt mất mép CCCD.
            if len(approx) == 4:
                padded_box = self.add_padding_to_box(
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    image_shape=resized_image.shape,
                )
                return padded_box, mask, contours

            # Nếu không đủ 4 điểm, dùng bounding rectangle có padding.
            fallback = self.add_padding_to_box(
                x=x,
                y=y,
                w=w,
                h=h,
                image_shape=resized_image.shape,
            )

            return fallback, mask, contours

        return None, mask, contours