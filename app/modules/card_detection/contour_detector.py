from __future__ import annotations

import math

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

    @staticmethod
    def _angle_distance(first: float, second: float) -> float:
        """Khoảng cách nhỏ nhất giữa hai hướng đường thẳng (0..90 độ)."""
        return abs((float(first) - float(second) + 90.0) % 180.0 - 90.0)

    @staticmethod
    def _order_quadrilateral(points: np.ndarray) -> np.ndarray:
        """Sắp bốn giao điểm theo chu vi, bắt đầu từ góc trên-trái."""
        quad = np.asarray(points, dtype=np.float32).reshape(4, 2)
        center = quad.mean(axis=0)
        angles = np.arctan2(
            quad[:, 1] - center[1],
            quad[:, 0] - center[0],
        )
        ordered = quad[np.argsort(angles)]
        start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
        return np.roll(ordered, -start, axis=0)

    @staticmethod
    def _line_intersection(
        first: dict[str, float],
        second: dict[str, float],
    ) -> np.ndarray | None:
        matrix = np.array(
            [
                [math.cos(first["theta"]), math.sin(first["theta"])],
                [math.cos(second["theta"]), math.sin(second["theta"])],
            ],
            dtype=np.float64,
        )
        if abs(float(np.linalg.det(matrix))) < 0.10:
            return None
        values = np.array(
            [first["rho"], second["rho"]],
            dtype=np.float64,
        )
        return np.linalg.solve(matrix, values).astype(np.float32)

    @staticmethod
    def _build_card_edge_map(image: np.ndarray) -> np.ndarray:
        """Giữ viền thẻ cả khi threshold sáng bị dính vào tay/nền."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        median_value = float(np.median(gray))
        lower = int(max(24.0, median_value * 0.34))
        upper = int(min(210.0, max(lower + 45.0, median_value * 1.05)))
        return cv2.Canny(
            gray,
            lower,
            upper,
            apertureSize=3,
            L2gradient=True,
        )

    @staticmethod
    def _edge_support(
        dilated_edges: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
    ) -> float:
        height, width = dilated_edges.shape[:2]
        length = float(np.linalg.norm(end - start))
        sample_count = max(12, int(round(length)))
        positions = np.linspace(0.0, 1.0, sample_count)
        x_values = np.rint(
            start[0] + (end[0] - start[0]) * positions
        ).astype(np.int32)
        y_values = np.rint(
            start[1] + (end[1] - start[1]) * positions
        ).astype(np.int32)
        x_values = np.clip(x_values, 0, width - 1)
        y_values = np.clip(y_values, 0, height - 1)
        return float(np.mean(dilated_edges[y_values, x_values] > 0))

    def _score_edge_quadrilateral(
        self,
        quadrilateral: np.ndarray,
        edge_map: np.ndarray,
        rank_bonus: float = 0.0,
        dilated_edges: np.ndarray | None = None,
    ) -> dict[str, float | list[float]] | None:
        height, width = edge_map.shape[:2]
        image_area = float(height * width)
        quad = self._order_quadrilateral(quadrilateral)
        area = abs(float(cv2.contourArea(quad)))
        area_ratio = area / max(image_area, 1.0)
        # Vùng nhỏ hơn 7% khung hình hầu như luôn là QR, chân dung hoặc
        # một cụm chữ bên trong thẻ. Ở kích thước detector cao 700 px, thẻ
        # nhỏ hơn mức này cũng không còn đủ chi tiết để OCR đáng tin cậy.
        if not 0.070 <= area_ratio <= 0.94:
            return None

        side_lengths = [
            float(np.linalg.norm(quad[(index + 1) % 4] - quad[index]))
            for index in range(4)
        ]
        first_axis = (side_lengths[0] + side_lengths[2]) * 0.5
        second_axis = (side_lengths[1] + side_lengths[3]) * 0.5
        short_axis = max(min(first_axis, second_axis), 1.0)
        long_axis = max(first_axis, second_axis)
        aspect_ratio = long_axis / short_axis
        if not 1.28 <= aspect_ratio <= 2.45:
            return None

        minimum_edge = min(side_lengths)
        if minimum_edge < max(55.0, min(height, width) * 0.10):
            return None

        opposite_balance = (
            min(side_lengths[0], side_lengths[2])
            / max(side_lengths[0], side_lengths[2], 1.0)
            * min(side_lengths[1], side_lengths[3])
            / max(side_lengths[1], side_lengths[3], 1.0)
        )
        if opposite_balance < 0.28:
            return None

        if dilated_edges is None:
            edge_radius = max(2, int(round(min(height, width) * 0.0045)))
            kernel_size = edge_radius * 2 + 1
            dilated_edges = cv2.dilate(
                edge_map,
                np.ones((kernel_size, kernel_size), dtype=np.uint8),
            )
        supports = [
            self._edge_support(
                dilated_edges,
                quad[index],
                quad[(index + 1) % 4],
            )
            for index in range(4)
        ]
        mean_support = float(np.mean(supports))
        minimum_support = float(min(supports))

        target_ratio = 85.60 / 53.98
        aspect_score = math.exp(
            -abs(math.log(aspect_ratio / target_ratio)) * 5.0
        )
        # Cạnh của QR/cụm chữ thường rõ hơn viền nhựa, nên điểm diện tích
        # phải đủ lớn để ưu tiên hình chữ nhật bao trọn thẻ.
        area_bonus = min(area_ratio / 0.38, 1.0) * 1.15

        interior_mask = np.zeros_like(edge_map)
        cv2.fillConvexPoly(
            interior_mask,
            np.rint(quad).astype(np.int32),
            255,
        )
        erosion_size = max(3, int(round(minimum_edge * 0.045)))
        interior_mask = cv2.erode(
            interior_mask,
            np.ones((erosion_size, erosion_size), dtype=np.uint8),
        )
        interior_pixels = int(np.count_nonzero(interior_mask))
        interior_edge_density = (
            float(np.count_nonzero(edge_map[interior_mask > 0]))
            / max(interior_pixels, 1)
        )
        content_bonus = 0.30 if 0.006 <= interior_edge_density <= 0.30 else -0.30

        # Một CCCD đầy đủ có chi tiết trải trên cả logo/chân dung bên trái,
        # khối chữ giữa và QR/phần nền bên phải. Chia ảnh đã nắn thành 4x3
        # ô giúp loại vùng nội bộ dù bốn cạnh của vùng đó rất rõ.
        destination = np.asarray(
            [[0, 0], [319, 0], [319, 201], [0, 201]],
            dtype=np.float32,
        )
        content_warp = cv2.warpPerspective(
            edge_map,
            cv2.getPerspectiveTransform(quad.astype(np.float32), destination),
            (320, 202),
        )
        cell_densities: list[float] = []
        for row in range(3):
            row_start = 10 + row * 60
            row_end = min(192, row_start + 60)
            for column in range(4):
                column_start = 10 + column * 75
                column_end = min(310, column_start + 75)
                cell = content_warp[
                    row_start:row_end,
                    column_start:column_end,
                ]
                cell_densities.append(float(np.mean(cell > 0)))

        content_grid_coverage = float(np.mean(
            np.asarray(cell_densities, dtype=np.float32) >= 0.016
        ))
        row_densities = [
            float(np.mean(cell_densities[row * 4:(row + 1) * 4]))
            for row in range(3)
        ]
        column_densities = [
            float(np.mean(cell_densities[column::4]))
            for column in range(4)
        ]
        content_distribution_bonus = (
            content_grid_coverage * 0.65
            + min(min(row_densities) / 0.055, 1.0) * 0.25
            + min(min(column_densities) / 0.045, 1.0) * 0.25
        )

        score = (
            aspect_score * 3.0
            + mean_support * 4.0
            + minimum_support * 1.5
            + opposite_balance
            + area_bonus
            + content_bonus
            + content_distribution_bonus
            + max(0.0, min(float(rank_bonus), 0.25))
        )
        return {
            "score": round(float(score), 4),
            "areaRatio": round(area_ratio, 4),
            "aspectRatio": round(aspect_ratio, 4),
            "oppositeBalance": round(opposite_balance, 4),
            "edgeSupportMean": round(mean_support, 4),
            "edgeSupportMinimum": round(minimum_support, 4),
            "interiorEdgeDensity": round(interior_edge_density, 4),
            "contentGridCoverage": round(content_grid_coverage, 4),
            "contentRowMinimum": round(min(row_densities), 4),
            "contentColumnMinimum": round(min(column_densities), 4),
            "edgeSupports": [round(value, 4) for value in supports],
        }

    def find_hough_card_quadrilaterals(
        self,
        image: np.ndarray,
        maximum_candidates: int = 3,
        edge_map: np.ndarray | None = None,
    ) -> tuple[list[dict[str, object]], np.ndarray]:
        """Dựng bốn góc từ hai cặp viền song song.

        Fallback này dành cho ảnh chụp xa: thẻ vẫn có bốn cạnh rõ nhưng
        threshold vùng sáng nối thẻ với bàn tay, màn hình hoặc nền sáng.
        """
        if edge_map is None:
            edge_map = self._build_card_edge_map(image)
        height, width = edge_map.shape[:2]
        diagonal = max(float(math.hypot(width, height)), 1.0)
        center_x = (width - 1) * 0.5
        center_y = (height - 1) * 0.5
        threshold = max(55, int(round(min(height, width) * 0.16)))
        raw_lines = cv2.HoughLines(
            edge_map,
            rho=1,
            theta=np.pi / 360.0,
            threshold=threshold,
        )
        if raw_lines is None:
            return [], edge_map

        lines: list[dict[str, float]] = []
        for rank, raw_line in enumerate(raw_lines[:320, 0, :]):
            rho = float(raw_line[0])
            theta = float(raw_line[1])
            direction = (math.degrees(theta) + 90.0) % 180.0
            normal_x = math.cos(theta)
            normal_y = math.sin(theta)
            # OpenCV có thể biểu diễn hai cạnh song song bằng hai pháp tuyến
            # ngược dấu (theta gần 0 và gần 180 độ). Chuẩn hóa dấu trước khi
            # đo khoảng cách; nếu không, hai mép thật của thẻ trông như chỉ
            # cách nhau vài pixel và bị loại khỏi danh sách cặp cạnh.
            normal_sign = -1.0 if (
                normal_x < -1e-6
                or (abs(normal_x) <= 1e-6 and normal_y < 0.0)
            ) else 1.0
            center_offset = (
                rho * normal_sign
                - center_x * normal_x * normal_sign
                - center_y * normal_y * normal_sign
            )
            lines.append({
                "rho": rho,
                "theta": theta,
                "direction": direction,
                "centerOffset": center_offset,
                "rank": float(rank),
            })

        def canonical_base(direction: float) -> float:
            value = direction % 90.0
            return value - 90.0 if value > 45.0 else value

        bases: list[float] = []
        for line in lines[:56]:
            base = canonical_base(line["direction"])
            if all(abs(base - existing) > 4.0 for existing in bases):
                bases.append(base)
            if len(bases) >= 10:
                break

        offset_tolerance = max(4.0, min(height, width) * 0.013)

        def deduplicate(group: list[dict[str, float]]) -> list[dict[str, float]]:
            selected: list[dict[str, float]] = []
            for line in group:
                duplicate = any(
                    self._angle_distance(
                        line["direction"],
                        existing["direction"],
                    ) < 1.75
                    and abs(
                        line["centerOffset"]
                        - existing["centerOffset"]
                    ) < offset_tolerance
                    for existing in selected
                )
                if duplicate:
                    continue
                selected.append(line)
                if len(selected) >= 20:
                    break
            return selected

        def build_pairs(
            group: list[dict[str, float]],
        ) -> list[tuple[float, dict[str, float], dict[str, float]]]:
            pairs: list[
                tuple[float, dict[str, float], dict[str, float]]
            ] = []
            minimum_separation = min(height, width) * 0.11
            maximum_separation = diagonal * 0.92
            for index, first in enumerate(group):
                for second in group[index + 1:]:
                    separation = abs(
                        first["centerOffset"] - second["centerOffset"]
                    )
                    if not minimum_separation <= separation <= maximum_separation:
                        continue
                    if self._angle_distance(
                        first["direction"],
                        second["direction"],
                    ) > 9.0:
                        continue
                    rank_sum = first["rank"] + second["rank"]
                    pairs.append((rank_sum, first, second))
            pairs.sort(key=lambda item: item[0])
            return pairs[:56]

        preliminary: list[tuple[float, float, np.ndarray]] = []
        seen_geometry: set[tuple[int, ...]] = set()
        target_ratio = 85.60 / 53.98

        for base in bases:
            first_group = deduplicate([
                line
                for line in lines
                if self._angle_distance(
                    line["direction"],
                    base % 180.0,
                ) < 10.0
            ])
            second_group = deduplicate([
                line
                for line in lines
                if self._angle_distance(
                    line["direction"],
                    (base + 90.0) % 180.0,
                ) < 10.0
            ])
            first_pairs = build_pairs(first_group)
            second_pairs = build_pairs(second_group)

            for first_rank, first_a, first_b in first_pairs:
                for second_rank, second_a, second_b in second_pairs:
                    intersections = [
                        self._line_intersection(first_a, second_a),
                        self._line_intersection(first_a, second_b),
                        self._line_intersection(first_b, second_b),
                        self._line_intersection(first_b, second_a),
                    ]
                    if any(point is None for point in intersections):
                        continue
                    quad = self._order_quadrilateral(
                        np.asarray(intersections, dtype=np.float32)
                    )
                    if (
                        np.any(quad[:, 0] < -3.0)
                        or np.any(quad[:, 0] > width + 2.0)
                        or np.any(quad[:, 1] < -3.0)
                        or np.any(quad[:, 1] > height + 2.0)
                        or not cv2.isContourConvex(
                            np.rint(quad).astype(np.int32)
                        )
                    ):
                        continue

                    side_lengths = [
                        float(np.linalg.norm(
                            quad[(index + 1) % 4] - quad[index]
                        ))
                        for index in range(4)
                    ]
                    axes = sorted([
                        (side_lengths[0] + side_lengths[2]) * 0.5,
                        (side_lengths[1] + side_lengths[3]) * 0.5,
                    ])
                    aspect_ratio = axes[1] / max(axes[0], 1.0)
                    area_ratio = abs(float(cv2.contourArea(quad))) / max(
                        float(height * width),
                        1.0,
                    )
                    if not (
                        0.070 <= area_ratio <= 0.94
                        and 1.28 <= aspect_ratio <= 2.45
                    ):
                        continue

                    key = tuple(
                        int(round(value / offset_tolerance))
                        for value in quad.reshape(-1)
                    )
                    if key in seen_geometry:
                        continue
                    seen_geometry.add(key)

                    opposite_balance = (
                        min(side_lengths[0], side_lengths[2])
                        / max(side_lengths[0], side_lengths[2], 1.0)
                        * min(side_lengths[1], side_lengths[3])
                        / max(side_lengths[1], side_lengths[3], 1.0)
                    )
                    aspect_score = math.exp(
                        -abs(math.log(aspect_ratio / target_ratio)) * 5.0
                    )
                    rank_sum = first_rank + second_rank
                    rank_bonus = max(0.0, 0.25 - rank_sum / 2400.0)
                    preliminary_score = (
                        aspect_score * 3.0
                        + opposite_balance
                        + min(area_ratio / 0.38, 1.0) * 1.15
                        + rank_bonus
                    )
                    preliminary.append(
                        (preliminary_score, rank_bonus, quad)
                    )

        preliminary.sort(key=lambda item: item[0], reverse=True)
        scored: list[dict[str, object]] = []
        edge_radius = max(2, int(round(min(height, width) * 0.0045)))
        kernel_size = edge_radius * 2 + 1
        dilated_edges = cv2.dilate(
            edge_map,
            np.ones((kernel_size, kernel_size), dtype=np.uint8),
        )
        for _, rank_bonus, quad in preliminary[:160]:
            metrics = self._score_edge_quadrilateral(
                quad,
                edge_map,
                rank_bonus=rank_bonus,
                dilated_edges=dilated_edges,
            )
            if metrics is None or float(metrics["score"]) < 7.10:
                continue
            scored.append({
                "corners": quad.reshape(4, 1, 2).astype(np.float32),
                **metrics,
            })

        scored.sort(
            key=lambda item: float(item["score"]),
            reverse=True,
        )
        selected: list[dict[str, object]] = []
        for candidate in scored:
            corners = np.asarray(candidate["corners"], dtype=np.float32)
            if any(
                float(np.mean(np.linalg.norm(
                    corners.reshape(4, 2)
                    - np.asarray(existing["corners"], dtype=np.float32).reshape(4, 2),
                    axis=1,
                ))) / diagonal < 0.025
                for existing in selected
            ):
                continue
            selected.append(candidate)
            if len(selected) >= maximum_candidates:
                break

        return selected, edge_map

    def find_card_contour_candidates_from_image(
        self,
        resized_image: np.ndarray,
    ) -> tuple[
        np.ndarray | None,
        np.ndarray,
        list[np.ndarray],
        dict[str, object],
    ]:
        """Trả quad chính cùng tối đa một quad Hough đã được kiểm chứng.

        Hough chỉ chạy khi contour sáng bị thiếu hoặc chạm biên đáng ngờ.
        Khi đã có contour trọn thẻ, bỏ qua Hough vừa tránh chọn QR/vùng chữ,
        vừa loại phần tính toán tổ hợp đường thẳng không cần thiết.
        """
        mask = self.build_card_mask(resized_image)
        contours = self.find_contours(mask)

        contour_quad: np.ndarray | None = None
        for contour in contours:
            if self.is_valid_card_contour(contour, resized_image.shape):
                contour_quad = self.contour_to_quadrilateral(
                    contour,
                    resized_image.shape,
                )
                break

        height, width = resized_image.shape[:2]
        image_area = max(float(height * width), 1.0)
        diagonal = max(float(math.hypot(width, height)), 1.0)
        edge_map = self._build_card_edge_map(resized_image)

        def area_ratio(points: np.ndarray) -> float:
            ordered = self._order_quadrilateral(points)
            return abs(float(cv2.contourArea(ordered))) / image_area

        def edge_touch_count(points: np.ndarray) -> int:
            ordered = self._order_quadrilateral(points)
            tolerance = max(4.0, min(height, width) * 0.012)
            return sum((
                float(np.min(ordered[:, 0])) <= tolerance,
                float(np.max(ordered[:, 0])) >= width - 1 - tolerance,
                float(np.min(ordered[:, 1])) <= tolerance,
                float(np.max(ordered[:, 1])) >= height - 1 - tolerance,
            ))

        primary_area_ratio = (
            area_ratio(contour_quad) if contour_quad is not None else 0.0
        )
        primary_edge_touches = (
            edge_touch_count(contour_quad) if contour_quad is not None else 0
        )
        primary_metrics: dict[str, object] = {}
        if contour_quad is not None:
            metrics = self._score_edge_quadrilateral(
                contour_quad,
                edge_map,
            )
            if metrics:
                primary_metrics = dict(metrics)

        primary_edge_score = float(primary_metrics.get("score") or 0.0)
        contour_is_complete = bool(
            contour_quad is not None
            and (
                primary_edge_touches == 0
                or primary_area_ratio >= 0.70
                or (
                    primary_edge_touches <= 1
                    and primary_edge_score >= 8.40
                )
            )
        )

        hough_candidates: list[dict[str, object]] = []
        hough_skipped_reason: str | None = None
        if contour_is_complete:
            hough_skipped_reason = "PRIMARY_CONTOUR_COMPLETE"
        else:
            hough_candidates, edge_map = self.find_hough_card_quadrilaterals(
                resized_image,
                maximum_candidates=6,
                edge_map=edge_map,
            )

        if contour_quad is None and not hough_candidates:
            return None, mask, contours, {
                "detectionMethod": "not_found",
                "houghFallbackEvaluated": True,
                "alternateCandidates": [],
            }

        def corner_distance(first: np.ndarray, second: np.ndarray) -> float:
            ordered_first = self._order_quadrilateral(first)
            ordered_second = self._order_quadrilateral(second)
            return float(np.mean(np.linalg.norm(
                ordered_first - ordered_second,
                axis=1,
            ))) / diagonal

        primary = contour_quad
        method = "brightness_contour"
        if hough_candidates:
            best_hough = hough_candidates[0]
            hough_corners = np.asarray(
                best_hough["corners"],
                dtype=np.float32,
            )
            if contour_quad is None:
                primary = hough_corners
                method = "hough_quadrilateral"
                primary_metrics = {
                    key: value
                    for key, value in best_hough.items()
                    if key != "corners"
                }
                primary_area_ratio = float(
                    best_hough.get("areaRatio") or area_ratio(hough_corners)
                )
                primary_edge_touches = edge_touch_count(hough_corners)
                mask = edge_map

        assert primary is not None
        alternates: list[dict[str, object]] = []
        if contour_quad is not None:
            ordered_contour = self._order_quadrilateral(contour_quad)
            contour_center = ordered_contour.mean(axis=0)
            contour_area = max(
                abs(float(cv2.contourArea(ordered_contour))),
                1.0,
            )
            for index, candidate in enumerate(hough_candidates, start=1):
                corners = np.asarray(candidate["corners"], dtype=np.float32)
                ordered_candidate = self._order_quadrilateral(corners)
                candidate_area = abs(float(cv2.contourArea(ordered_candidate)))
                relative_area = candidate_area / contour_area
                intersection_area, _ = cv2.intersectConvexConvex(
                    ordered_contour.astype(np.float32),
                    ordered_candidate.astype(np.float32),
                )
                overlap = float(intersection_area) / max(candidate_area, 1.0)
                center_distance = float(np.linalg.norm(
                    ordered_candidate.mean(axis=0) - contour_center
                )) / diagonal
                grid_coverage = float(
                    candidate.get("contentGridCoverage") or 0.0
                )

                # Hình chữ nhật dự phòng phải có kích thước tương đương vùng
                # thẻ chính và gần như nằm trọn trong vùng đó. Điều kiện này
                # loại QR/chân dung/cụm chữ nhưng vẫn cho phép Hough cắt bỏ
                # phần tay hoặc nền bị contour sáng dính vào.
                if not (
                    0.48 <= relative_area <= 1.15
                    and overlap >= 0.82
                    and center_distance <= 0.22
                    and grid_coverage >= 0.83
                    and candidate_area / image_area >= 0.12
                ):
                    continue
                if corner_distance(primary, corners) < 0.025:
                    continue

                detection = {
                    key: value
                    for key, value in candidate.items()
                    if key != "corners"
                }
                detection.update({
                    "relativeToPrimaryArea": round(relative_area, 4),
                    "overlapWithPrimary": round(overlap, 4),
                    "centerDistanceFromPrimary": round(center_distance, 4),
                    "wholeCardCandidate": True,
                })
                alternates.append({
                    "name": f"hough_whole_card_{index}",
                    "corners": corners,
                    "detection": detection,
                })
                # Chỉ OCR một phương án Hough đã qua kiểm tra. Nhiều phương
                # án gần nhau làm tăng thời gian nhưng không thêm thông tin.
                break

        whole_card_reliable = bool(
            (
                method == "hough_quadrilateral"
                and float(
                    primary_metrics.get("contentGridCoverage") or 0.0
                ) >= 0.83
                and float(primary_metrics.get("edgeSupportMean") or 0.0)
                >= 0.62
            )
            or contour_is_complete
        )

        return primary, mask, contours, {
            "detectionMethod": method,
            "detectionScore": primary_metrics.get("score"),
            "detectionMetrics": primary_metrics,
            "primaryAreaRatio": round(primary_area_ratio, 4),
            "primaryEdgeTouchCount": primary_edge_touches,
            "wholeCardReliable": whole_card_reliable,
            "houghFallbackEvaluated": not contour_is_complete,
            "houghSkippedReason": hough_skipped_reason,
            "alternateCandidates": alternates[:1],
        }

    def find_card_contour_from_image(
        self,
        resized_image: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray, list[np.ndarray]]:
        contour, mask, contours, _ = (
            self.find_card_contour_candidates_from_image(resized_image)
        )
        return contour, mask, contours
