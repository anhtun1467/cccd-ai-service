from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2

from app.core.config import settings
from app.modules.card_detection.detector import CardDetector
from app.modules.ocr.field_ocr_service import field_ocr_service
from app.modules.ocr.line_merger import OCRLineMerger
from app.modules.ocr.result_fuser import fuse_ocr_data
from app.modules.ocr.service import ocr_service
from app.modules.ocr.text_normalizer import OCRTextNormalizer
from app.modules.ocr.validator import CCCDValidator


class OcrPipelineService:
    """
    Điều phối toàn bộ pipeline OCR CCCD.

    Quy trình:
        Upload image
        -> Card Detection
        -> Perspective Transform
        -> Enhancement
        -> OCR toàn thẻ
        -> OCR từng vùng
        -> Hợp nhất kết quả
        -> Validator
        -> JSON response
    """

    FIELD_NAMES: tuple[str, ...] = (
        "idNumber",
        "fullName",
        "dateOfBirth",
        "gender",
        "nationality",
        "placeOfOrigin",
        "placeOfResidence",
        "dateOfExpiry",
    )

    def __init__(self) -> None:
        self.card_detector = CardDetector()
        self.ocr_service = ocr_service
        self.field_ocr_service = field_ocr_service
        self.validator = CCCDValidator()

        self.line_merger = OCRLineMerger(
            vertical_tolerance_ratio=0.6,
            maximum_horizontal_gap_ratio=3.5,
        )

    def process_cccd_image(
        self,
        image_path: str,
    ) -> dict[str, Any]:
        """
        Xử lý một ảnh CCCD và trả về dữ liệu có cấu trúc.
        """

        start_time = time.perf_counter()

        image_file = Path(image_path)

        if not image_file.exists():
            return self.build_error_response(
                message=(
                    f"Không tìm thấy ảnh đầu vào: "
                    f"{image_file}"
                ),
                start_time=start_time,
            )

        if not image_file.is_file():
            return self.build_error_response(
                message=(
                    f"Đường dẫn không phải tệp ảnh: "
                    f"{image_file}"
                ),
                start_time=start_time,
            )

        file_stem = image_file.stem

        output_dir = Path(settings.output_dir)
        debug_dir = (
            Path("storage")
            / "debug"
            / file_stem
        )
        field_output_dir = (
            debug_dir
            / "fields"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        debug_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            detection_result = (
                self.card_detector.detect_from_path(
                    image_path=str(image_file),
                    output_dir=str(debug_dir),
                )
            )
        except Exception as error:
            return self.build_error_response(
                message=(
                    "Phát hiện vùng CCCD thất bại: "
                    f"{error}"
                ),
                start_time=start_time,
            )

        card_image = detection_result.get(
            "cardImage"
        )
        enhanced_image = detection_result.get(
            "enhancedImage"
        )

        if card_image is None:
            return self.build_error_response(
                message=(
                    "Không phát hiện được vùng CCCD "
                    "trong ảnh"
                ),
                start_time=start_time,
            )

        if enhanced_image is None:
            return self.build_error_response(
                message=(
                    "Không tạo được ảnh CCCD "
                    "đã tăng cường"
                ),
                start_time=start_time,
            )

        card_output_path = (
            output_dir
            / f"{file_stem}_card.jpg"
        )

        enhanced_output_path = (
            output_dir
            / f"{file_stem}_enhanced.jpg"
        )

        card_saved = cv2.imwrite(
            str(card_output_path),
            card_image,
        )

        enhanced_saved = cv2.imwrite(
            str(enhanced_output_path),
            enhanced_image,
        )

        if not card_saved:
            return self.build_error_response(
                message=(
                    "Không thể lưu ảnh CCCD: "
                    f"{card_output_path}"
                ),
                start_time=start_time,
            )

        if not enhanced_saved:
            return self.build_error_response(
                message=(
                    "Không thể lưu ảnh CCCD "
                    f"đã tăng cường: {enhanced_output_path}"
                ),
                start_time=start_time,
            )

        full_ocr_result = self.run_full_card_ocr(
            enhanced_image_path=enhanced_output_path,
        )

        field_ocr_result = self.run_field_ocr(
            card_image_path=card_output_path,
            field_output_dir=field_output_dir,
        )

        full_card_data = self.make_json_safe(
            full_ocr_result.get(
                "structuredData",
                {},
            )
        )

        field_data = self.make_json_safe(
            field_ocr_result.get(
                "structuredData",
                {},
            )
        )

        raw_text_for_fusion = self.make_json_safe(
            full_ocr_result.get(
                "normalizedText",
                [],
            )
        )

        if isinstance(
            raw_text_for_fusion,
            str,
        ):
            raw_text_for_fusion = (
                raw_text_for_fusion.splitlines()
            )

        if not isinstance(
            raw_text_for_fusion,
            list,
        ):
            raw_text_for_fusion = []

        merged_data, data_sources = fuse_ocr_data(
            full_card_data=full_card_data,
            field_data=field_data,
            raw_text=raw_text_for_fusion,
        )

        validation_result = self.validator.validate(
            merged_data
        )

        text_boxes = self.make_json_safe(
            full_ocr_result.get(
                "textBoxes",
                [],
            )
        )

        merged_text_boxes = self.make_json_safe(
            full_ocr_result.get(
                "mergedTextBoxes",
                [],
            )
        )

        normalized_text = raw_text_for_fusion

        processing_time = round(
            time.perf_counter() - start_time,
            3,
        )

        average_confidence = (
            self.calculate_average_confidence(
                text_boxes
            )
        )

        field_confidences = (
            self.get_field_confidences(
                field_ocr_result
            )
        )

        portrait_result = self.make_json_safe(
            field_ocr_result.get(
                "portrait"
            )
        )

        field_results = self.make_json_safe(
            field_ocr_result.get(
                "fieldResults",
                {},
            )
        )

        field_debug = self.make_json_safe(
            field_ocr_result.get(
                "debug",
                {},
            )
        )

        return {
            "status": "OCR_SUCCESS",
            "message": "OCR CCCD thành công",
            "cccdData": merged_data,
            "metadata": {
                "engine": "EasyOCR",
                "processingTime": processing_time,
                "averageConfidence": (
                    average_confidence
                ),
                "fieldConfidences": (
                    field_confidences
                ),
                "validation": self.make_json_safe(
                    validation_result
                ),
                "inputImage": str(image_file),
                "cardImage": str(
                    card_output_path
                ),
                "enhancedImage": str(
                    enhanced_output_path
                ),
                "debugDir": str(debug_dir),
                "fieldDebug": field_debug,
                "resizeRatio": self.make_json_safe(
                    detection_result.get(
                        "resizeRatio",
                        1.0,
                    )
                ),
                "fullCardData": full_card_data,
                "fieldData": field_data,
                "dataSources": self.make_json_safe(
                    data_sources
                ),
            },
            "portrait": portrait_result,
            "rawText": normalized_text,
            "textBoxes": text_boxes,
            "mergedTextBoxes": (
                merged_text_boxes
            ),
            "fieldResults": field_results,
        }

    def run_full_card_ocr(
        self,
        enhanced_image_path: Path,
    ) -> dict[str, Any]:
        """
        OCR toàn bộ ảnh CCCD.
        """

        try:
            result = (
                self.ocr_service.extract_cccd_info(
                    str(enhanced_image_path)
                )
            )

            if not result:
                return self.empty_full_ocr_result(
                    message=(
                        "OCR toàn thẻ không trả về "
                        "kết quả"
                    )
                )

            return result

        except Exception as error:
            return self.empty_full_ocr_result(
                message=(
                    "OCR toàn thẻ thất bại: "
                    f"{error}"
                )
            )

    def run_field_ocr(
        self,
        card_image_path: Path,
        field_output_dir: Path,
    ) -> dict[str, Any]:
        """
        Cắt và OCR từng field trên CCCD.
        """

        try:
            result = (
                self.field_ocr_service.extract_fields(
                    card_image_path=str(
                        card_image_path
                    ),
                    output_dir=str(
                        field_output_dir
                    ),
                )
            )

            if not result:
                return self.empty_field_ocr_result(
                    message=(
                        "OCR từng vùng không trả về "
                        "kết quả"
                    )
                )

            return result

        except Exception as error:
            return self.empty_field_ocr_result(
                message=(
                    "OCR từng vùng thất bại: "
                    f"{error}"
                )
            )

    def merge_structured_data(
        self,
        field_data: dict[str, Any],
        full_card_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Hợp nhất dữ liệu OCR.

        Ưu tiên:
        1. Kết quả OCR từng field.
        2. Kết quả parser OCR toàn thẻ.
        3. None.
        """

        merged_data: dict[str, Any] = {}

        for field_name in self.FIELD_NAMES:
            field_value = self.normalize_value(
                field_data.get(
                    field_name
                )
            )

            full_card_value = (
                self.normalize_value(
                    full_card_data.get(
                        field_name
                    )
                )
            )

            if self.is_valid_value(
                field_value
            ):
                merged_data[field_name] = (
                    field_value
                )

            elif self.is_valid_value(
                full_card_value
            ):
                merged_data[field_name] = (
                    full_card_value
                )

            else:
                merged_data[field_name] = None

        return merged_data

    def resolve_data_sources(
        self,
        field_data: dict[str, Any],
        full_card_data: dict[str, Any],
    ) -> dict[str, str]:
        """
        Cho biết mỗi trường dữ liệu đến từ nguồn nào.
        """

        sources: dict[str, str] = {}

        for field_name in self.FIELD_NAMES:
            field_value = self.normalize_value(
                field_data.get(
                    field_name
                )
            )

            full_card_value = (
                self.normalize_value(
                    full_card_data.get(
                        field_name
                    )
                )
            )

            if self.is_valid_value(
                field_value
            ):
                sources[field_name] = (
                    "FIELD_OCR"
                )

            elif self.is_valid_value(
                full_card_value
            ):
                sources[field_name] = (
                    "FULL_CARD_OCR"
                )

            else:
                sources[field_name] = (
                    "NOT_FOUND"
                )

        return sources

    @staticmethod
    def normalize_value(
        value: Any,
    ) -> Any:
        """
        Chuẩn hóa giá trị trước khi hợp nhất.
        """

        if value is None:
            return None

        if isinstance(value, str):
            cleaned = value.strip()

            if not cleaned:
                return None

            return cleaned

        return value

    @staticmethod
    def is_valid_value(
        value: Any,
    ) -> bool:
        """
        Kiểm tra giá trị có thể sử dụng hay không.
        """

        if value is None:
            return False

        if isinstance(value, str):
            invalid_values = {
                "",
                "none",
                "null",
                "unknown",
                "not found",
            }

            return (
                value.strip().lower()
                not in invalid_values
            )

        return True

    @staticmethod
    def get_field_confidences(
        field_ocr_result: dict[str, Any],
    ) -> dict[str, float]:
        """
        Lấy confidence của từng field.
        """

        field_results = (
            field_ocr_result.get(
                "fieldResults",
                {},
            )
        )

        confidences: dict[str, float] = {}

        if not isinstance(
            field_results,
            dict,
        ):
            return confidences

        for field_name, result in (
            field_results.items()
        ):
            if not isinstance(result, dict):
                continue

            try:
                confidence = float(
                    result.get(
                        "averageConfidence",
                        0.0,
                    )
                )
            except (TypeError, ValueError):
                confidence = 0.0

            confidences[field_name] = round(
                confidence,
                4,
            )

        return confidences

    @staticmethod
    def calculate_average_confidence(
        text_boxes: list[dict[str, Any]],
    ) -> float:
        """
        Tính confidence trung bình OCR toàn thẻ.
        """

        if not text_boxes:
            return 0.0

        scores: list[float] = []

        for item in text_boxes:
            if not isinstance(item, dict):
                continue

            try:
                confidence = float(
                    item.get(
                        "confidence",
                        0.0,
                    )
                )
            except (TypeError, ValueError):
                continue

            if 0.0 <= confidence <= 1.0:
                scores.append(confidence)

        if not scores:
            return 0.0

        return round(
            sum(scores) / len(scores),
            4,
        )

    def empty_full_ocr_result(
        self,
        message: str,
    ) -> dict[str, Any]:
        """
        Kết quả mặc định khi OCR toàn thẻ lỗi.
        """

        return {
            "ocrSuccess": False,
            "ocrMessage": message,
            "structuredData": {
                field_name: None
                for field_name
                in self.FIELD_NAMES
            },
            "validation": {
                "isValid": False,
                "errors": [message],
            },
            "normalizedText": [],
            "textBoxes": [],
            "mergedTextBoxes": [],
        }

    def empty_field_ocr_result(
        self,
        message: str,
    ) -> dict[str, Any]:
        """
        Kết quả mặc định khi OCR từng vùng lỗi.
        """

        return {
            "structuredData": {
                field_name: None
                for field_name
                in self.FIELD_NAMES
            },
            "fieldResults": {
                field_name: {
                    "fieldName": field_name,
                    "success": False,
                    "message": message,
                    "value": None,
                    "averageConfidence": 0.0,
                }
                for field_name
                in self.FIELD_NAMES
            },
            "portrait": None,
            "debug": {},
        }

    def build_error_response(
        self,
        message: str,
        start_time: float,
    ) -> dict[str, Any]:
        """
        Tạo response khi pipeline thất bại.
        """

        processing_time = round(
            time.perf_counter()
            - start_time,
            3,
        )

        return {
            "status": "OCR_FAILED",
            "message": message,
            "cccdData": {
                field_name: None
                for field_name
                in self.FIELD_NAMES
            },
            "metadata": {
                "engine": "EasyOCR",
                "processingTime": (
                    processing_time
                ),
                "averageConfidence": 0.0,
                "fieldConfidences": {},
                "validation": {
                    "isValid": False,
                    "errors": [message],
                },
            },
            "portrait": None,
            "rawText": [],
            "textBoxes": [],
            "mergedTextBoxes": [],
            "fieldResults": {},
        }

    def make_json_safe(
        self,
        value: Any,
    ) -> Any:
        """
        Chuyển dữ liệu NumPy và các kiểu đặc biệt
        sang dạng JSON-safe.
        """

        if isinstance(value, dict):
            return {
                str(key): self.make_json_safe(
                    item
                )
                for key, item
                in value.items()
            }

        if isinstance(value, list):
            return [
                self.make_json_safe(item)
                for item in value
            ]

        if isinstance(value, tuple):
            return [
                self.make_json_safe(item)
                for item in value
            ]

        if isinstance(value, Path):
            return str(value)

        if hasattr(value, "tolist"):
            return self.make_json_safe(
                value.tolist()
            )

        if hasattr(value, "item"):
            return value.item()

        return value


ocr_pipeline_service = OcrPipelineService()
