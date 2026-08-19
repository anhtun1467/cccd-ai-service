from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np


class FastCardOrientation:
    """Xác định chiều 0/180 độ của mặt trước CCCD trước khi chạy OCR.

    Bộ phân loại chỉ khóa chiều khi có tín hiệu mạnh: QR nằm đúng một trong
    hai góc hợp lệ, cụm màu đỏ đặc trưng của quốc huy/tiêu đề, hoặc các finder
    pattern lồng nhau của QR. Trường hợp không chắc chắn được trả về UNKNOWN
    để pipeline OCR cũ tiếp tục làm fallback an toàn.
    """

    MAXIMUM_DIMENSION = 720
    MINIMUM_RED_PIXEL_RATIO = 0.0012
    MAXIMUM_RED_PIXEL_RATIO = 0.12

    def analyze(
        self,
        image: np.ndarray | None,
        qr_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        result: dict[str, Any] = {
            "reliable": False,
            "rotationDegrees": 0,
            "source": "UNKNOWN",
            "confidence": 0.0,
            "reason": "INSUFFICIENT_VISUAL_EVIDENCE",
            "signals": {},
            "elapsedMs": 0.0,
        }
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            result["reason"] = "INVALID_IMAGE"
            return self._finish(result, started)

        height, width = image.shape[:2]
        qr_signal = self._qr_position_evidence(
            qr_result or {},
            card_size=(width, height),
        )
        result["signals"]["qrPosition"] = qr_signal
        if qr_signal.get("reliable"):
            result.update({
                "reliable": True,
                "rotationDegrees": int(qr_signal["rotationDegrees"]),
                "source": str(qr_signal["source"]),
                "confidence": float(qr_signal["confidence"]),
                "reason": str(qr_signal["reason"]),
            })
            return self._finish(result, started)

        reduced = self._limit_image_size(image)
        red_signal = self._red_layout_evidence(reduced)
        finder_signal = self._qr_finder_evidence(reduced)
        result["signals"]["redLayout"] = red_signal
        result["signals"]["qrFinderPattern"] = finder_signal

        reliable_signals = [
            signal
            for signal in (red_signal, finder_signal)
            if signal.get("reliable")
        ]
        if len(reliable_signals) >= 2 and len({
            int(signal["rotationDegrees"])
            for signal in reliable_signals
        }) > 1:
            result["reason"] = "VISUAL_SIGNALS_CONFLICT"
            return self._finish(result, started)

        if reliable_signals:
            selected = max(
                reliable_signals,
                key=lambda signal: float(signal.get("confidence", 0.0)),
            )
            result.update({
                "reliable": True,
                "rotationDegrees": int(selected["rotationDegrees"]),
                "source": str(selected["source"]),
                "confidence": float(selected["confidence"]),
                "reason": str(selected["reason"]),
            })

        return self._finish(result, started)

    @staticmethod
    def _finish(
        result: dict[str, Any],
        started: float,
    ) -> dict[str, Any]:
        result["elapsedMs"] = round(
            (time.perf_counter() - started) * 1000.0,
            2,
        )
        return result

    @classmethod
    def _limit_image_size(cls, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        maximum = max(height, width)
        if maximum <= cls.MAXIMUM_DIMENSION:
            return image
        scale = cls.MAXIMUM_DIMENSION / float(maximum)
        return cv2.resize(
            image,
            (
                max(1, round(width * scale)),
                max(1, round(height * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def _qr_position_evidence(
        qr_result: dict[str, Any],
        card_size: tuple[int, int],
    ) -> dict[str, Any]:
        signal: dict[str, Any] = {
            "reliable": False,
            "rotationDegrees": 0,
            "source": "QR_POSITION",
            "confidence": 0.0,
            "reason": "QR_POSITION_UNAVAILABLE",
        }
        if not isinstance(qr_result, dict):
            return signal

        card_width = max(int(card_size[0]), 1)
        card_height = max(int(card_size[1]), 1)
        box = qr_result.get("boundingBox")
        if not isinstance(box, dict):
            return signal
        try:
            x = float(box.get("x", 0.0))
            y = float(box.get("y", 0.0))
            width = float(box.get("width", 0.0))
            height = float(box.get("height", 0.0))
        except (TypeError, ValueError):
            return signal
        if width <= 0.0 or height <= 0.0:
            return signal

        square_ratio = max(width, height) / max(min(width, height), 1.0)
        width_ratio = width / float(card_width)
        height_ratio = height / float(card_height)
        area_ratio = width * height / float(card_width * card_height)
        center_x = (x + width * 0.5) / float(card_width)
        center_y = (y + height * 0.5) / float(card_height)
        signal.update({
            "center": [round(center_x, 4), round(center_y, 4)],
            "sizeRatio": [round(width_ratio, 4), round(height_ratio, 4)],
            "areaRatio": round(area_ratio, 4),
            "squareRatio": round(square_ratio, 4),
        })

        plausible_shape = bool(
            square_ratio <= 1.48
            and 0.045 <= width_ratio <= 0.34
            and 0.065 <= height_ratio <= 0.42
            and 0.003 <= area_ratio <= 0.13
        )
        top_right = bool(center_x >= 0.60 and center_y <= 0.44)
        bottom_left = bool(center_x <= 0.40 and center_y >= 0.56)
        if not plausible_shape or not (top_right or bottom_left):
            signal["reason"] = "QR_REGION_NOT_IN_ORIENTATION_CORNER"
            return signal

        decoded = bool(qr_result.get("decoded"))
        region_detected = bool(
            qr_result.get("regionDetected")
            or qr_result.get("qrRegionDetected")
        )
        if not decoded and not region_detected:
            return signal

        signal.update({
            "reliable": True,
            "rotationDegrees": 180 if bottom_left else 0,
            "source": (
                "CCCD_QR_POSITION" if decoded else "QR_REGION_POSITION"
            ),
            "confidence": 0.99 if decoded else 0.90,
            "reason": (
                "QR_BOTTOM_LEFT" if bottom_left else "QR_TOP_RIGHT"
            ),
        })
        return signal

    @classmethod
    def _red_layout_evidence(cls, image: np.ndarray) -> dict[str, Any]:
        signal: dict[str, Any] = {
            "reliable": False,
            "rotationDegrees": 0,
            "source": "FRONT_RED_LAYOUT",
            "confidence": 0.0,
            "reason": "RED_LAYOUT_NOT_DISTINCT",
        }
        if image.ndim != 3 or image.shape[2] < 3:
            signal["reason"] = "COLOR_IMAGE_REQUIRED"
            return signal

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blue, green, red = cv2.split(image[:, :, :3])
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        hue_red = (hue <= 12) | (hue >= 168)
        red_dominant = (
            red.astype(np.float32) >= green.astype(np.float32) * 1.18
        ) & (
            red.astype(np.float32) >= blue.astype(np.float32) * 1.15
        )
        mask = (
            hue_red
            & (saturation >= 65)
            & (value >= 55)
            & red_dominant
        ).astype(np.uint8) * 255
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
        )

        height, width = mask.shape[:2]
        margin_y = max(1, round(height * 0.015))
        margin_x = max(1, round(width * 0.015))
        interior = mask[
            margin_y:max(margin_y + 1, height - margin_y),
            margin_x:max(margin_x + 1, width - margin_x),
        ]
        ys, _xs = np.nonzero(interior)
        pixel_count = len(ys)
        pixel_ratio = pixel_count / float(max(interior.size, 1))
        signal["redPixelRatio"] = round(pixel_ratio, 5)
        if not (
            cls.MINIMUM_RED_PIXEL_RATIO
            <= pixel_ratio
            <= cls.MAXIMUM_RED_PIXEL_RATIO
        ):
            signal["reason"] = "RED_PIXEL_RATIO_OUT_OF_RANGE"
            return signal

        normalized_y = (
            float(np.mean(ys) + margin_y) / float(max(height - 1, 1))
        )
        top_count = int(np.count_nonzero(mask[: height // 2]))
        bottom_count = int(np.count_nonzero(mask[height // 2 :]))
        total = max(top_count + bottom_count, 1)
        top_fraction = top_count / float(total)
        bottom_fraction = bottom_count / float(total)
        dominance = abs(top_fraction - bottom_fraction)
        signal.update({
            "centroidY": round(normalized_y, 4),
            "topFraction": round(top_fraction, 4),
            "bottomFraction": round(bottom_fraction, 4),
            "verticalDominance": round(dominance, 4),
        })

        upright = bool(top_fraction >= 0.68 and normalized_y <= 0.46)
        upside_down = bool(bottom_fraction >= 0.68 and normalized_y >= 0.54)
        if not (upright or upside_down):
            return signal

        confidence = min(0.97, 0.60 + dominance * 0.45)
        signal.update({
            "reliable": True,
            "rotationDegrees": 180 if upside_down else 0,
            "confidence": round(confidence, 4),
            "reason": (
                "RED_LAYOUT_BOTTOM_HALF"
                if upside_down
                else "RED_LAYOUT_TOP_HALF"
            ),
        })
        return signal

    @classmethod
    def _qr_finder_evidence(cls, image: np.ndarray) -> dict[str, Any]:
        signal: dict[str, Any] = {
            "reliable": False,
            "rotationDegrees": 0,
            "source": "QR_FINDER_PATTERN",
            "confidence": 0.0,
            "reason": "QR_FINDER_PATTERN_NOT_DISTINCT",
        }
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image.ndim == 3
            else image.copy()
        )
        height, width = gray.shape[:2]
        if height < 80 or width < 120:
            signal["reason"] = "IMAGE_TOO_SMALL"
            return signal

        regions = {
            "topRight": gray[
                0 : round(height * 0.48),
                round(width * 0.60) : width,
            ],
            "bottomLeft": gray[
                round(height * 0.52) : height,
                0 : round(width * 0.40),
            ],
        }

        def nested_square_score(region: np.ndarray) -> float:
            if region.size == 0:
                return 0.0
            blurred = cv2.GaussianBlur(region, (3, 3), 0)
            binary = cv2.threshold(
                blurred,
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )[1]
            contours, hierarchy = cv2.findContours(
                binary,
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if hierarchy is None:
                return 0.0
            hierarchy_rows = hierarchy[0]
            region_area = float(region.shape[0] * region.shape[1])
            score = 0.0
            for index, contour in enumerate(contours):
                x, y, box_width, box_height = cv2.boundingRect(contour)
                del x, y
                if min(box_width, box_height) < 5:
                    continue
                square_ratio = max(box_width, box_height) / float(
                    max(min(box_width, box_height), 1)
                )
                box_area_ratio = box_width * box_height / max(region_area, 1.0)
                if square_ratio > 1.32 or not 0.0008 <= box_area_ratio <= 0.18:
                    continue
                rectangle_area = float(box_width * box_height)
                if float(cv2.contourArea(contour)) / max(rectangle_area, 1.0) < 0.42:
                    continue

                depth = 0
                child = int(hierarchy_rows[index][2])
                visited: set[int] = set()
                while child >= 0 and child not in visited and depth < 5:
                    visited.add(child)
                    depth += 1
                    child = int(hierarchy_rows[child][2])
                if depth >= 2:
                    score += min(float(depth), 4.0)
            return score

        top_score = nested_square_score(regions["topRight"])
        bottom_score = nested_square_score(regions["bottomLeft"])
        signal.update({
            "topRightScore": round(top_score, 2),
            "bottomLeftScore": round(bottom_score, 2),
        })
        winner = max(top_score, bottom_score)
        difference = abs(top_score - bottom_score)
        if winner < 4.0 or difference < 2.5:
            return signal

        upside_down = bottom_score > top_score
        signal.update({
            "reliable": True,
            "rotationDegrees": 180 if upside_down else 0,
            "confidence": round(
                min(0.92, 0.68 + difference / max(winner, 1.0) * 0.24),
                4,
            ),
            "reason": (
                "QR_FINDER_BOTTOM_LEFT"
                if upside_down
                else "QR_FINDER_TOP_RIGHT"
            ),
        })
        return signal
