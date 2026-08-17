from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from typing import Any

import cv2
import numpy as np

from app.modules.qr.cccd_qr_parser import CCCDQRParser
from app.modules.qr.diagnostics import (
    build_qr_error_details,
    derive_qr_status,
    qr_status_message,
)

try:
    import zxingcpp
except ImportError:  # pragma: no cover - OpenCV vẫn là fallback bắt buộc.
    zxingcpp = None


PointMapper = Callable[[np.ndarray], np.ndarray]


class CCCDQRDecoder:
    """Khoanh và giải mã QR CCCD với fallback hữu hạn.

    ZXing-C++ được ưu tiên vì QR CCCD có mật độ dữ liệu cao và OpenCV có
    thể khoanh đúng nhưng vẫn không giải mã được. OpenCV vẫn được giữ để
    dự án chạy khi dependency tùy chọn chưa được cài và để tạo vùng debug.
    """

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
        result = self.empty_result()
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            result["errors"] = ["QR_IMAGE_INVALID"]
            return self.finalize_result(result, started)

        original_height, original_width = image.shape[:2]
        limited = self.limit_image_size(image, maximum_dimension=1600)
        limited_height, limited_width = limited.shape[:2]
        scale_x = original_width / float(max(limited_width, 1))
        scale_y = original_height / float(max(limited_height, 1))
        result["searchRegion"] = self.build_search_region(
            original_width,
            original_height,
        )

        invalid_payload_errors: list[str] = []
        payload_was_decoded = False

        if zxingcpp is not None and result["attemptCount"] < self.maximum_attempts:
            result["attemptCount"] += 1
            try:
                barcodes = zxingcpp.read_barcodes(
                    limited,
                    formats=zxingcpp.BarcodeFormat.QRCode,
                    try_rotate=False,
                    try_downscale=True,
                    return_errors=True,
                )
            except Exception:
                barcodes = []
                result["errors"].append("QR_DECODE_ERROR:zxing_card_raw")

            for barcode in barcodes:
                polygon = self.zxing_polygon(barcode)
                if polygon is not None:
                    polygon[:, 0] *= scale_x
                    polygon[:, 1] *= scale_y
                    self.record_region(
                        result,
                        polygon,
                        variant_name="card_raw",
                        image=image,
                    )

                payload = ""
                try:
                    if bool(barcode.valid):
                        payload = str(barcode.text or "")
                except Exception:
                    payload = ""
                if not payload:
                    continue

                payload_was_decoded = True
                result["payloadDecoded"] = True
                parsed = self.parser.parse(payload)
                if self.apply_parsed_payload(
                    result,
                    parsed,
                    selected_variant="card_raw",
                    selected_decoder="ZXing-C++",
                ):
                    return self.finalize_result(result, started)
                invalid_payload_errors.extend(parsed.get("errors", []))
                break

        if not payload_was_decoded:
            detector = cv2.QRCodeDetector()
            for record in self.iter_variant_records(image):
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if (
                    result["attemptCount"] > 0
                    and elapsed_ms >= self.time_budget_ms
                ):
                    result["errors"].append("QR_TIME_BUDGET_REACHED")
                    break
                if result["attemptCount"] >= self.maximum_attempts:
                    break

                variant_name = str(record["name"])
                variant_image = record["image"]
                mapper: PointMapper = record["mapPoints"]
                result["attemptCount"] += 1
                try:
                    detected, points = detector.detect(variant_image)
                except cv2.error:
                    result["errors"].append(
                        f"QR_DECODE_ERROR:{variant_name}"
                    )
                    continue

                if not detected or points is None:
                    continue

                mapped_points = mapper(
                    np.asarray(points, dtype=np.float32).reshape(-1, 2)
                )
                self.record_region(
                    result,
                    mapped_points,
                    variant_name=variant_name,
                    image=image,
                )
                try:
                    decode_output = detector.decode(variant_image, points)
                    payload = str(decode_output[0] or "")
                except (cv2.error, IndexError, TypeError):
                    payload = ""
                if not payload:
                    continue

                payload_was_decoded = True
                result["payloadDecoded"] = True
                parsed = self.parser.parse(payload)
                if self.apply_parsed_payload(
                    result,
                    parsed,
                    selected_variant=variant_name,
                    selected_decoder="OpenCV QRCodeDetector",
                ):
                    return self.finalize_result(result, started)
                invalid_payload_errors.extend(parsed.get("errors", []))
                break

        if not result["decoded"] and invalid_payload_errors:
            result["errors"].append("NON_CCCD_QR_IGNORED")
            result["errors"].extend(sorted(set(invalid_payload_errors)))

        if not result["decoded"]:
            if result["regionDetected"] and not result["payloadDecoded"]:
                result["errors"].append("QR_REGION_DETECTED_NOT_DECODED")
            elif not result["regionDetected"]:
                result["errors"].append("QR_REGION_NOT_DETECTED")

        return self.finalize_result(result, started)

    @staticmethod
    def empty_result() -> dict[str, Any]:
        return {
            "decoded": False,
            "used": False,
            "payloadDecoded": False,
            "decoder": "ZXing-C++ + OpenCV",
            "selectedDecoder": None,
            "attemptCount": 0,
            "selectedVariant": None,
            "detectionVariant": None,
            "elapsedMs": 0.0,
            "format": None,
            "fieldCount": 0,
            "providedFields": [],
            "missingRequiredFields": [],
            "structuredData": {},
            "auxiliaryData": {},
            "regionDetected": False,
            "qrRegionDetected": False,
            "polygon": [],
            "boundingBox": None,
            "searchRegion": None,
            "debugCrop": None,
            "status": "NOT_DETECTED",
            "statusMessage": "",
            "errors": [],
            "errorDetails": [],
        }

    def finalize_result(
        self,
        result: dict[str, Any],
        started: float,
    ) -> dict[str, Any]:
        deduplicated_errors: list[str] = []
        for error in result.get("errors", []):
            text = str(error or "").strip()
            if text and text not in deduplicated_errors:
                deduplicated_errors.append(text)
        result["errors"] = deduplicated_errors
        result["status"] = derive_qr_status(result)
        result["statusMessage"] = qr_status_message(result["status"])
        result["errorDetails"] = build_qr_error_details(
            deduplicated_errors
        )
        result["elapsedMs"] = round(
            (time.perf_counter() - started) * 1000.0,
            2,
        )
        return result

    @staticmethod
    def apply_parsed_payload(
        result: dict[str, Any],
        parsed: dict[str, Any],
        selected_variant: str,
        selected_decoder: str,
    ) -> bool:
        if not parsed.get("success"):
            result["missingRequiredFields"] = list(
                parsed.get("missingRequiredFields", []) or []
            )
            return False
        result.update(
            {
                "decoded": True,
                "used": True,
                "payloadDecoded": True,
                "selectedDecoder": selected_decoder,
                "selectedVariant": selected_variant,
                "format": parsed.get("format"),
                "fieldCount": int(parsed.get("fieldCount", 0) or 0),
                "providedFields": list(
                    parsed.get("providedFields", []) or []
                ),
                "missingRequiredFields": list(
                    parsed.get("missingRequiredFields", []) or []
                ),
                "structuredData": dict(
                    parsed.get("structuredData", {}) or {}
                ),
                "auxiliaryData": dict(
                    parsed.get("auxiliaryData", {}) or {}
                ),
                "errors": list(parsed.get("errors", []) or []),
            }
        )
        return True

    @staticmethod
    def zxing_polygon(barcode: Any) -> np.ndarray | None:
        try:
            position = barcode.position
            points = (
                position.top_left,
                position.top_right,
                position.bottom_right,
                position.bottom_left,
            )
            return np.asarray(
                [[float(point.x), float(point.y)] for point in points],
                dtype=np.float32,
            )
        except Exception:
            return None

    @staticmethod
    def record_region(
        result: dict[str, Any],
        polygon: np.ndarray,
        variant_name: str,
        image: np.ndarray,
    ) -> None:
        if result.get("regionDetected"):
            return
        height, width = image.shape[:2]
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(points) < 4:
            return
        points[:, 0] = np.clip(points[:, 0], 0, max(width - 1, 0))
        points[:, 1] = np.clip(points[:, 1], 0, max(height - 1, 0))
        x1 = int(np.floor(np.min(points[:, 0])))
        y1 = int(np.floor(np.min(points[:, 1])))
        x2 = int(np.ceil(np.max(points[:, 0])))
        y2 = int(np.ceil(np.max(points[:, 1])))
        if x2 <= x1 or y2 <= y1:
            return
        padding = max(4, int(round(min(x2 - x1, y2 - y1) * 0.08)))
        crop_x1 = max(0, x1 - padding)
        crop_y1 = max(0, y1 - padding)
        crop_x2 = min(width, x2 + padding + 1)
        crop_y2 = min(height, y2 + padding + 1)
        result.update(
            {
                "regionDetected": True,
                "qrRegionDetected": True,
                "detectionVariant": variant_name,
                "polygon": [
                    [round(float(x), 2), round(float(y), 2)]
                    for x, y in points[:4]
                ],
                "boundingBox": {
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1,
                },
                "debugCrop": image[crop_y1:crop_y2, crop_x1:crop_x2].copy(),
            }
        )

    @staticmethod
    def build_search_region(width: int, height: int) -> dict[str, int]:
        x1 = max(0, int(round(width * 0.60)))
        y2 = min(height, int(round(height * 0.48)))
        return {
            "x": x1,
            "y": 0,
            "width": max(0, width - x1),
            "height": max(0, y2),
        }

    def iter_variant_records(
        self,
        image: np.ndarray,
    ) -> Iterator[dict[str, Any]]:
        original_height, original_width = image.shape[:2]
        card = self.limit_image_size(image, maximum_dimension=1600)
        height, width = card.shape[:2]
        base_scale_x = original_width / float(max(width, 1))
        base_scale_y = original_height / float(max(height, 1))

        x1 = max(0, int(round(width * 0.60)))
        y2 = min(height, int(round(height * 0.48)))
        roi = card[0:y2, x1:width]

        if roi.size:
            enlarged, roi_scale, border = self.enlarge_qr_roi_with_metadata(
                roi
            )

            def map_roi_points(points: np.ndarray) -> np.ndarray:
                mapped = np.asarray(points, dtype=np.float32).copy()
                mapped[:, 0] = (
                    (mapped[:, 0] - border) / roi_scale + x1
                ) * base_scale_x
                mapped[:, 1] = (
                    (mapped[:, 1] - border) / roi_scale
                ) * base_scale_y
                return mapped

            yield {
                "name": "top_right_raw",
                "image": enlarged,
                "mapPoints": map_roi_points,
            }
            gray = (
                cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
                if enlarged.ndim == 3
                else enlarged.copy()
            )
            yield {
                "name": "top_right_gray",
                "image": gray,
                "mapPoints": map_roi_points,
            }
            clahe = cv2.createCLAHE(
                clipLimit=1.8,
                tileGridSize=(8, 8),
            ).apply(gray)
            blurred = cv2.GaussianBlur(clahe, (0, 0), sigmaX=0.8)
            sharpened = cv2.addWeighted(clahe, 1.35, blurred, -0.35, 0)
            yield {
                "name": "top_right_detail",
                "image": sharpened,
                "mapPoints": map_roi_points,
            }
            binary = cv2.threshold(
                sharpened,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )[1]
            yield {
                "name": "top_right_otsu",
                "image": binary,
                "mapPoints": map_roi_points,
            }

        def map_card_points(points: np.ndarray) -> np.ndarray:
            mapped = np.asarray(points, dtype=np.float32).copy()
            mapped[:, 0] *= base_scale_x
            mapped[:, 1] *= base_scale_y
            return mapped

        yield {
            "name": "card_raw",
            "image": card,
            "mapPoints": map_card_points,
        }

    def iter_variants(
        self,
        image: np.ndarray,
    ) -> Iterator[tuple[str, np.ndarray]]:
        """Giữ API cũ cho benchmark/test bên ngoài dự án."""
        for record in self.iter_variant_records(image):
            yield str(record["name"]), record["image"]

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
            (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def enlarge_qr_roi_with_metadata(
        image: np.ndarray,
    ) -> tuple[np.ndarray, float, int]:
        height, width = image.shape[:2]
        target_maximum = 720
        scale = max(
            1.0,
            min(5.0, target_maximum / float(max(height, width))),
        )
        resized = image
        if scale > 1.0:
            resized = cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
        border = max(12, int(round(min(resized.shape[:2]) * 0.05)))
        border_value: int | tuple[int, int, int]
        border_value = 255 if resized.ndim == 2 else (255, 255, 255)
        bordered = cv2.copyMakeBorder(
            resized,
            border,
            border,
            border,
            border,
            cv2.BORDER_CONSTANT,
            value=border_value,
        )
        return bordered, scale, border

    @staticmethod
    def enlarge_qr_roi(image: np.ndarray) -> np.ndarray:
        enlarged, _, _ = CCCDQRDecoder.enlarge_qr_roi_with_metadata(image)
        return enlarged
