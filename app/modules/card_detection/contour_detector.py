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
        multi_card_min_area_ratio: float = 0.12,
        multi_card_min_rectangularity: float = 0.68,
    ):
        self.max_contours = max_contours
        self.epsilon_factor = epsilon_factor
        self.min_area_ratio = min_area_ratio
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.padding = padding
        self.multi_card_min_area_ratio = multi_card_min_area_ratio
        self.multi_card_min_rectangularity = multi_card_min_rectangularity

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

    def build_multi_card_mask(self, resized_image: np.ndarray) -> np.ndarray:
        """Tạo mask hạt mịn để không nối hai thẻ đặt sát cạnh nhau.

        Mask chính dùng kernel 25x25 nhằm vá viền thẻ đơn. Với ảnh có hai
        CCCD, kernel đó có thể lấp luôn khe ở giữa và biến hai thẻ thành một
        contour lớn. Mask này chỉ open nhẹ 3x3 để giữ nguyên đường phân cách.
        """
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, threshold = cv2.threshold(
            blur,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(
            threshold,
            cv2.MORPH_OPEN,
            kernel,
        )

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

    @staticmethod
    def _overlap_over_smaller_area(
        first: np.ndarray,
        second: np.ndarray,
    ) -> float:
        first_x, first_y, first_w, first_h = cv2.boundingRect(first)
        second_x, second_y, second_w, second_h = cv2.boundingRect(second)

        intersection_width = max(
            0,
            min(first_x + first_w, second_x + second_w)
            - max(first_x, second_x),
        )
        intersection_height = max(
            0,
            min(first_y + first_h, second_y + second_h)
            - max(first_y, second_y),
        )
        intersection = float(intersection_width * intersection_height)
        smaller = float(min(first_w * first_h, second_w * second_h))
        return intersection / smaller if smaller > 0 else 0.0

    def find_multiple_card_contours_from_image(
        self,
        resized_image: np.ndarray,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Tìm từ hai vùng CCCD độc lập để chặn ảnh ghép nhiều thẻ."""
        mask = self.build_multi_card_mask(resized_image)
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        image_height, image_width = resized_image.shape[:2]
        image_area = float(image_height * image_width)
        target_ratio = 85.60 / 53.98
        candidates: list[tuple[float, float, np.ndarray]] = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < image_area * self.multi_card_min_area_ratio:
                continue

            (_, _), (rect_width, rect_height), _ = cv2.minAreaRect(contour)
            short_edge = min(rect_width, rect_height)
            long_edge = max(rect_width, rect_height)
            if short_edge <= 0:
                continue

            aspect_ratio = long_edge / float(short_edge)
            if not (
                self.min_aspect_ratio
                <= aspect_ratio
                <= self.max_aspect_ratio
            ):
                continue

            rectangle_area = float(rect_width * rect_height)
            rectangularity = area / rectangle_area if rectangle_area > 0 else 0.0
            if rectangularity < self.multi_card_min_rectangularity:
                continue

            quadrilateral = self.contour_to_quadrilateral(
                contour,
                resized_image.shape,
            )
            candidates.append(
                (
                    abs(aspect_ratio - target_ratio),
                    -area,
                    quadrilateral,
                )
            )

        # Ưu tiên contour có tỷ lệ gần CCCD thật. Contour bao quanh cả hai
        # thẻ sẽ bị loại vì chồng gần như toàn bộ lên từng contour con.
        selected: list[np.ndarray] = []
        for _, _, candidate in sorted(candidates, key=lambda item: item[:2]):
            if any(
                self._overlap_over_smaller_area(candidate, chosen) >= 0.72
                for chosen in selected
            ):
                continue
            selected.append(candidate)

        if len(selected) < 2:
            tiled_cards = self.find_tiled_card_regions(resized_image)
            if len(tiled_cards) >= 2:
                return tiled_cards, mask
            return selected, mask

        total_area = sum(float(cv2.contourArea(item)) for item in selected)
        if total_area < image_area * 0.35:
            tiled_cards = self.find_tiled_card_regions(resized_image)
            if len(tiled_cards) >= 2:
                return tiled_cards, mask
            return selected[:1], mask

        return selected, mask

    def find_tiled_card_regions(
        self,
        image: np.ndarray,
    ) -> list[np.ndarray]:
        """Fallback cho hai thẻ đặt sát nhau làm contour bị dính.

        Hai CCCD dọc đặt cạnh nhau hoặc hai CCCD ngang xếp chồng tạo ra
        một đường phân cách tối liên tục gần giữa ảnh. Ngoài độ tối, hai
        nửa sau khi chia đều phải có tỷ lệ hình học giống thẻ ID-1; nhờ
        vậy đường chữ/ảnh chân dung trên một thẻ đơn không bị tính nhầm.
        """
        height, width = image.shape[:2]
        if height < 120 or width < 120:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        margin_y = max(1, int(round(height * 0.05)))
        margin_x = max(1, int(round(width * 0.05)))

        def valid_card_ratio(first_edge: int, second_edge: int) -> bool:
            short_edge = min(first_edge, second_edge)
            long_edge = max(first_edge, second_edge)
            if short_edge <= 0:
                return False
            ratio = long_edge / float(short_edge)
            return self.min_aspect_ratio <= ratio <= self.max_aspect_ratio

        # Trường hợp hai thẻ dọc đặt cạnh nhau.
        vertical_profile = np.mean(
            gray[margin_y:height - margin_y, :],
            axis=0,
        )
        search_left = int(round(width * 0.30))
        search_right = int(round(width * 0.70))
        if search_right > search_left:
            central = vertical_profile[search_left:search_right]
            split_x = search_left + int(np.argmin(central))
            contrast = float(np.median(central) - vertical_profile[split_x])
            left_width = split_x
            right_width = width - split_x
            if (
                contrast >= 18.0
                and min(left_width, right_width) >= width * 0.30
                and valid_card_ratio(height, left_width)
                and valid_card_ratio(height, right_width)
            ):
                gap = max(1, int(round(width * 0.002)))
                return [
                    np.array(
                        [[[0, 0]], [[split_x - gap, 0]],
                         [[split_x - gap, height - 1]], [[0, height - 1]]],
                        dtype=np.float32,
                    ),
                    np.array(
                        [[[split_x + gap, 0]], [[width - 1, 0]],
                         [[width - 1, height - 1]],
                         [[split_x + gap, height - 1]]],
                        dtype=np.float32,
                    ),
                ]

        # Trường hợp hai thẻ ngang xếp trên dưới.
        horizontal_profile = np.mean(
            gray[:, margin_x:width - margin_x],
            axis=1,
        )
        search_top = int(round(height * 0.30))
        search_bottom = int(round(height * 0.70))
        if search_bottom > search_top:
            central = horizontal_profile[search_top:search_bottom]
            split_y = search_top + int(np.argmin(central))
            contrast = float(np.median(central) - horizontal_profile[split_y])
            top_height = split_y
            bottom_height = height - split_y
            if (
                contrast >= 18.0
                and min(top_height, bottom_height) >= height * 0.30
                and valid_card_ratio(width, top_height)
                and valid_card_ratio(width, bottom_height)
            ):
                gap = max(1, int(round(height * 0.002)))
                return [
                    np.array(
                        [[[0, 0]], [[width - 1, 0]],
                         [[width - 1, split_y - gap]],
                         [[0, split_y - gap]]],
                        dtype=np.float32,
                    ),
                    np.array(
                        [[[0, split_y + gap]],
                         [[width - 1, split_y + gap]],
                         [[width - 1, height - 1]], [[0, height - 1]]],
                        dtype=np.float32,
                    ),
                ]

        return []

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
        """Tìm bốn góc thật và chọn xấp xỉ giữ được nhiều viền nhất."""
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        hull_area = max(float(cv2.contourArea(hull)), 1.0)
        contour_area = max(float(cv2.contourArea(contour)), 1.0)

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

        candidates: list[tuple[float, float, np.ndarray]] = []
        seen: set[tuple[int, ...]] = set()

        for factor in dict.fromkeys(factors):
            approx = cv2.approxPolyDP(
                hull,
                factor * perimeter,
                True,
            )
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue

            approx_area = float(cv2.contourArea(approx))
            if approx_area < contour_area * 0.75:
                continue

            # Chuẩn hóa thứ tự theo góc quanh tâm để loại ứng viên trùng
            # giữa nhiều epsilon khác nhau.
            points = approx.reshape(4, 2).astype(np.float32)
            center = points.mean(axis=0)
            order = np.argsort(
                np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
            )
            key = tuple(
                int(round(value))
                for value in points[order].reshape(-1)
            )
            if key in seen:
                continue
            seen.add(key)

            retained_hull = min(1.0, approx_area / hull_area)
            retained_contour = min(1.0, approx_area / contour_area)
            score = retained_hull * 0.70 + retained_contour * 0.30
            candidates.append(
                (
                    score,
                    -float(factor),
                    approx.astype(np.float32),
                )
            )

        if not candidates:
            return None

        # Bản cũ trả ứng viên đầu tiên (epsilon 3%), dù một epsilon khác có
        # thể giữ góc thật tốt hơn. Chọn theo diện tích viền được bảo toàn,
        # rồi ưu tiên epsilon nhỏ khi điểm bằng nhau.
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return candidates[0][2]

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
