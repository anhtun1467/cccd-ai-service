from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import cv2
import numpy as np

from app.modules.qr.cccd_qr_parser import CCCDQRParser


class CCCDQRDecoder:
    """Giải mã QR CCCD với số biến thể và ngân sách thời gian hữu hạn."""

    def __init__(
        self,
        time_budget_ms: float = 120.0,
        maximum_attempts: int = 5,
        parser: CCCDQRParser | None = None,
    ) -> None:
        self.time_budget_ms = max(20.0, float(time_budget_ms))
        self.maximum_attempts = max(1, int(maximum_attempts))
        self.parser = parser or CCCDQRParser()

    def decode(self, image: np.ndarray | None) -> dict[str, Any]:
        started = time.perf_counter()
        result: dict[str, Any] = {
            "decoded": False,
            "used": False,
            "decoder": "OpenCV QRCodeDetector",
            "attemptCount": 0,
            "selectedVariant": None,
            "elapsedMs": 0.0,
            "format": None,
            "fieldCount": 0,
            "providedFields": [],
            "structuredData": {},
            "auxiliaryData": {},
            "errors": [],
        }
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            result["errors"] = ["QR_IMAGE_INVALID"]
            return result

        detector = cv2.QRCodeDetector()
        invalid_payload_errors: list[str] = []
        for variant_name, variant_image in self.iter_variants(image):
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if (
                result["attemptCount"] > 0
                and elapsed_ms >= self.time_budget_ms
            ):
                result["errors"].append("QR_TIME_BUDGET_REACHED")
                break
            if result["attemptCount"] >= self.maximum_attempts:
                break

            result["attemptCount"] += 1
            try:
                payload, points, _ = detector.detectAndDecode(variant_image)
            except cv2.error:
                result["errors"].append(
                    f"QR_DECODE_ERROR:{variant_name}"
                )
                continue
            if not payload:
                continue

            parsed = self.parser.parse(payload)
            if not parsed.get("success"):
                invalid_payload_errors.extend(parsed.get("errors", []))
                # Ảnh thẻ chỉ được phép cung cấp một QR CCCD. Khi OpenCV
                # đã đọc rõ một QR khác loại, thử thêm các biến thể của
                # cùng vùng không thể biến payload đó thành CCCD.
                break

            result.update(
                {
                    "decoded": True,
                    "used": True,
                    "selectedVariant": variant_name,
                    "format": parsed.get("format"),
                    "fieldCount": int(parsed.get("fieldCount", 0) or 0),
                    "providedFields": list(
                        parsed.get("providedFields", [])
                    ),
                    "structuredData": dict(
                        parsed.get("structuredData", {})
                    ),
                    "auxiliaryData": dict(
                        parsed.get("auxiliaryData", {})
                    ),
                    "qrRegionDetected": points is not None,
                    "errors": list(parsed.get("errors", [])),
                }
            )
            break

        if not result["decoded"] and invalid_payload_errors:
            result["errors"].append("NON_CCCD_QR_IGNORED")
            result["errors"].extend(sorted(set(invalid_payload_errors)))
        result["elapsedMs"] = round(
            (time.perf_counter() - started) * 1000.0,
            2,
        )
        return result

    def iter_variants(
        self,
        image: np.ndarray,
    ) -> Iterator[tuple[str, np.ndarray]]:
        card = self.limit_image_size(image, maximum_dimension=1600)
        yield "card_raw", card

        height, width = card.shape[:2]
        x1 = max(0, int(round(width * 0.67)))
        y2 = min(height, int(round(height * 0.40)))
        roi = card[0:y2, x1:width]
        if roi.size == 0:
            return

        enlarged = self.enlarge_qr_roi(roi)
        yield "top_right_raw", enlarged

        gray = (
            cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
            if enlarged.ndim == 3
            else enlarged.copy()
        )
        yield "top_right_gray", gray

        clahe = cv2.createCLAHE(
            clipLimit=1.8,
            tileGridSize=(8, 8),
        ).apply(gray)
        blurred = cv2.GaussianBlur(clahe, (0, 0), sigmaX=0.8)
        sharpened = cv2.addWeighted(clahe, 1.35, blurred, -0.35, 0)
        yield "top_right_detail", sharpened

        binary = cv2.threshold(
            sharpened,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[1]
        yield "top_right_otsu", binary

    @staticmethod
    def limit_image_size(
        image: np.ndarray,
        maximum_dimension: int,
    ) -> np.ndarray:
        height, width = image.shape[:2]
        current_maximum = max(height, width)
        if current_maximum <= maximum_dimension:
            return image
        scale = maximum_dimension / float(current_maximum)
        return cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def enlarge_qr_roi(image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        target_maximum = 720
        scale = max(1.0, min(5.0, target_maximum / float(max(height, width))))
        if scale > 1.0:
            image = cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
        border = max(12, int(round(min(image.shape[:2]) * 0.05)))
        border_value: int | tuple[int, int, int]
        border_value = 255 if image.ndim == 2 else (255, 255, 255)
        return cv2.copyMakeBorder(
            image,
            border,
            border,
            border,
            border,
            cv2.BORDER_CONSTANT,
            value=border_value,
        )
