from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import cv2

from app.core.config import settings
from app.core.exceptions import BadRequestException
from app.modules.card_detection.detector import CardDetector
from app.modules.card_detection.image_enhancer import cccd_image_enhancer
from app.modules.ocr.field_ocr_service import field_ocr_service
from app.modules.ocr.line_merger import OCRLineMerger
from app.modules.ocr.result_fuser import (
    estimate_layout_y_offset,
    fuse_ocr_data,
    remove_accents,
)
from app.modules.ocr.service import ocr_service
from app.modules.ocr.validator import CCCDValidator
from app.utils.image_validator import check_image_quality


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
        -> Lưu JSON
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

    ORIENTATION_RETRY_THRESHOLD = 7.0
    ORIENTATION_SELECTION_MARGIN = 1.5
    CROP_BLUR_WARNING_THRESHOLD = 80.0
    CROP_DARK_WARNING_THRESHOLD = 60.0
    MINIMUM_READABLE_CORE_FIELDS = 3

    def __init__(self) -> None:
        self.card_detector = CardDetector()
        self.ocr_service = ocr_service
        self.field_ocr_service = field_ocr_service
        if getattr(self.field_ocr_service, "engine", None) is None:
            self.field_ocr_service.engine = self.ocr_service.engine
        self.validator = CCCDValidator()

        self.line_merger = OCRLineMerger(
            vertical_tolerance_ratio=0.35,
            maximum_horizontal_gap_ratio=1.8,
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
                image_file=image_file,
            )

        if not image_file.is_file():
            return self.build_error_response(
                message=(
                    f"Đường dẫn không phải tệp ảnh: "
                    f"{image_file}"
                ),
                start_time=start_time,
                image_file=image_file,
            )

        file_stem = image_file.stem

        output_dir = Path(settings.output_dir)
        debug_dir = Path("storage") / "debug" / file_stem
        json_output_dir = Path("storage") / "json"
        field_output_dir = debug_dir / "fields"

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        debug_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        json_output_dir.mkdir(
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
        except BadRequestException as error:
            rejection = dict(error.data or {})
            rejection.setdefault("errorCode", "CARD_DETECTION_FAILED")
            rejection.setdefault("reason", error.message)
            return self.build_error_response(
                message=error.message,
                start_time=start_time,
                image_file=image_file,
                rejection=rejection,
            )
        except Exception as error:
            return self.build_error_response(
                message=(
                    "Phát hiện vùng CCCD thất bại: "
                    f"{error}"
                ),
                start_time=start_time,
                image_file=image_file,
                rejection={
                    "errorCode": "CARD_DETECTION_FAILED",
                    "reason": "Không phát hiện được vùng CCCD",
                    "suggestion": (
                        "Vui lòng đặt trọn một CCCD trong khung hình."
                    ),
                },
            )

        card_image = detection_result.get(
            "cardImage"
        )

        detector_enhanced_image = detection_result.get(
            "enhancedImage"
        )

        if card_image is None:
            return self.build_error_response(
                message=(
                    "Không phát hiện được vùng CCCD "
                    "trong ảnh"
                ),
                start_time=start_time,
                image_file=image_file,
            )

        blur_info = cccd_image_enhancer.estimate_blur(
            card_image
        )

        # Laplacian thấp chỉ là cảnh báo. Ảnh đã perspective/resize thường
        # có điểm thấp hơn ảnh gốc dù chữ vẫn còn đọc được, vì vậy quyết
        # định từ chối được hoãn đến sau OCR và dựa thêm vào các trường lõi.
        cropped_quality = check_image_quality(
            card_image,
            blur_threshold=self.CROP_BLUR_WARNING_THRESHOLD,
            dark_threshold=self.CROP_DARK_WARNING_THRESHOLD,
        )

        if detector_enhanced_image is not None:
            enhanced_image = detector_enhanced_image
        else:
            enhanced_image = cccd_image_enhancer.enhance(
                card_image
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
                image_file=image_file,
            )

        if not enhanced_saved:
            return self.build_error_response(
                message=(
                    "Không thể lưu ảnh CCCD đã tăng cường: "
                    f"{enhanced_output_path}"
                ),
                start_time=start_time,
                image_file=image_file,
            )

        # OCR chính chạy trên ảnh warped gốc.
        full_ocr_result = self.run_full_card_ocr(
            image_path=card_output_path,
        )

        (
            card_image,
            enhanced_image,
            full_ocr_result,
            orientation_info,
        ) = self.ensure_upright_orientation(
            card_image=card_image,
            enhanced_image=enhanced_image,
            card_output_path=card_output_path,
            enhanced_output_path=enhanced_output_path,
            debug_dir=debug_dir,
            first_ocr_result=full_ocr_result,
        )

        full_ocr_variant = "card"
        if (
            cropped_quality["is_blurry"]
            or float(orientation_info.get("selectedScore", 0.0))
            < self.ORIENTATION_RETRY_THRESHOLD + 3.0
        ):
            enhanced_ocr_result = self.run_full_card_ocr(
                image_path=enhanced_output_path,
            )
            full_ocr_result, full_ocr_variant = (
                self.select_better_full_ocr_result(
                    card_result=full_ocr_result,
                    enhanced_result=enhanced_ocr_result,
                )
            )

        geometry_rotation = int(
            detection_result.get("geometry", {}).get(
                "geometryRotationDegrees",
                0,
            )
        )
        orientation_info["geometryRotationDegrees"] = geometry_rotation
        orientation_info["totalRotationDegrees"] = (
            geometry_rotation
            + int(orientation_info.get("contentRotationDegrees", 0))
        ) % 360

        layout_y_offset = estimate_layout_y_offset(
            full_ocr_result.get("textBoxes", []),
            image_size=(
                int(card_image.shape[1]),
                int(card_image.shape[0]),
            ),
        )
        orientation_info["layoutYOffset"] = layout_y_offset

        field_ocr_result = self.run_field_ocr(
            card_image_path=card_output_path,
            field_output_dir=field_output_dir,
            layout_y_offset=layout_y_offset,
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
                full_ocr_result.get(
                    "rawText",
                    [],
                ),
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

        print("=" * 80)
        print("RAW TEXT FOR FUSION")

        if raw_text_for_fusion:
            for index, line in enumerate(
                raw_text_for_fusion
            ):
                print(
                    f"{index}: {line!r}"
                )
        else:
            print(
                "[WARNING] raw_text_for_fusion "
                "đang rỗng"
            )

        print("=" * 80)

        merged_data, data_sources = fuse_ocr_data(
            full_card_data=full_card_data,
            field_data=field_data,
            raw_text=raw_text_for_fusion,
            field_results=field_ocr_result.get("fieldResults", {}),
            text_boxes=full_ocr_result.get("textBoxes", []),
            image_size=(
                int(card_image.shape[1]),
                int(card_image.shape[0]),
            ),
        )

        validation_result = self.validator.validate(
            merged_data
        )

        readability = self.evaluate_ocr_readability(
            merged_data=merged_data,
            cropped_quality=cropped_quality,
            raw_text=raw_text_for_fusion,
        )

        if not readability["isReadable"]:
            return self.build_error_response(
                message=(
                    "OCR không thu được đủ thông tin cốt lõi từ vùng CCCD"
                ),
                start_time=start_time,
                image_file=image_file,
                rejection={
                    "errorCode": "OCR_UNREADABLE_IMAGE",
                    "reason": (
                        "Ảnh chưa cung cấp đủ bằng chứng OCR để kết luận"
                    ),
                    "blurScore": cropped_quality["blur_score"],
                    "brightnessScore": (
                        cropped_quality["brightness_score"]
                    ),
                    "readableCoreFields": readability[
                        "readableCoreFields"
                    ],
                    "suggestion": (
                        "Vui lòng lấy nét vào phần số CCCD, họ tên và "
                        "ngày sinh rồi chụp lại."
                    ),
                },
                partial_result={
                    "cccdData": merged_data,
                    "validation": validation_result,
                    "rawText": raw_text_for_fusion,
                    "textBoxes": full_ocr_result.get("textBoxes", []),
                    "mergedTextBoxes": full_ocr_result.get(
                        "mergedTextBoxes",
                        [],
                    ),
                    "fieldResults": field_ocr_result.get(
                        "fieldResults",
                        {},
                    ),
                    "fieldConfidences": self.get_field_confidences(
                        field_ocr_result
                    ),
                },
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

        json_output_path = (
            json_output_dir
            / f"{file_stem}.json"
        )

        is_fully_valid = bool(validation_result.get("isValid"))
        response = {
            "status": (
                "OCR_SUCCESS" if is_fully_valid else "OCR_PARTIAL"
            ),
            "message": (
                "OCR CCCD thành công"
                if is_fully_valid
                else "OCR hoàn tất nhưng có trường cần kiểm tra"
            ),
            "cccdData": merged_data,
            "metadata": {
                "engine": "EasyOCR",
                "processingTime": processing_time,
                "averageConfidence": (
                    average_confidence
                ),
                "imageQuality": self.make_json_safe({
                    **blur_info,
                    "cropBlurScore": cropped_quality["blur_score"],
                    "brightnessScore": cropped_quality[
                        "brightness_score"
                    ],
                    "decision": (
                        "REVIEW_REQUIRED"
                        if not is_fully_valid
                        else (
                            "PASSED_WITH_WARNING"
                            if not cropped_quality["is_valid"]
                            else "PASSED"
                        )
                    ),
                    "warnings": readability["warnings"],
                    "readableCoreFields": readability[
                        "readableCoreFields"
                    ],
                }),
                "fieldConfidences": (
                    field_confidences
                ),
                "validation": self.make_json_safe(
                    validation_result
                ),
                "reviewRequired": not is_fully_valid,
                "inputImage": str(image_file),
                "cardImage": str(
                    card_output_path
                ),
                "enhancedImage": str(
                    enhanced_output_path
                ),
                "debugDir": str(debug_dir),
                "jsonOutput": str(
                    json_output_path
                ),
                "fieldDebug": field_debug,
                "geometry": self.make_json_safe(
                    detection_result.get("geometry", {})
                ),
                "orientation": self.make_json_safe(
                    orientation_info
                ),
                "fullCardOcrVariant": full_ocr_variant,
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

        self.save_json_response(
            response=response,
            json_path=json_output_path,
        )

        return response

    def run_full_card_ocr(
        self,
        image_path: Path,
    ) -> dict[str, Any]:
        """
        OCR toàn bộ ảnh CCCD.
        """

        try:
            result = (
                self.ocr_service.extract_cccd_info(
                    str(image_path)
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

    def score_full_ocr_result(
        self,
        ocr_result: dict[str, Any],
    ) -> float:
        """Chấm một lần OCR theo nhãn, trường lõi và confidence."""
        score = self.calculate_orientation_score(ocr_result)
        structured = ocr_result.get("structuredData", {})
        if not isinstance(structured, dict):
            structured = {}

        if self.validator.is_valid_id_number(structured.get("idNumber")):
            score += 4.0
        if self.validator.is_valid_name(structured.get("fullName")):
            score += 2.0
        if self.validator.is_valid_date(structured.get("dateOfBirth")):
            score += 2.0
        if self.validator.is_valid_gender(structured.get("gender")):
            score += 1.0
        if self.validator.is_valid_nationality(
            structured.get("nationality")
        ):
            score += 1.0

        confidence = self.calculate_average_confidence(
            self.make_json_safe(ocr_result.get("textBoxes", []))
        )
        score += confidence * 2.0
        return round(float(score), 3)

    def select_better_full_ocr_result(
        self,
        card_result: dict[str, Any],
        enhanced_result: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        """Chọn ảnh gốc hoặc ảnh tăng cường bằng chất lượng OCR thực tế."""
        card_score = self.score_full_ocr_result(card_result)
        enhanced_score = self.score_full_ocr_result(enhanced_result)

        if enhanced_score > card_score + 0.75:
            return enhanced_result, "enhanced"
        return card_result, "card"

    def evaluate_ocr_readability(
        self,
        merged_data: dict[str, Any],
        cropped_quality: dict[str, Any],
        raw_text: list[str],
    ) -> dict[str, Any]:
        """
        Chỉ từ chối khi điểm ảnh kém *và* OCR không đủ trường lõi.

        Cách này tránh coi ảnh hơi mờ là lỗi tuyệt đối, đồng thời không trả
        JSON có vẻ hợp lệ khi OCR chỉ thu được vài chuỗi rời rạc.
        """
        field_checks = {
            "idNumber": self.validator.is_valid_id_number(
                merged_data.get("idNumber")
            ),
            "fullName": self.validator.is_valid_name(
                merged_data.get("fullName")
            ),
            "dateOfBirth": self.validator.is_valid_date(
                merged_data.get("dateOfBirth")
            ),
            "gender": self.validator.is_valid_gender(
                merged_data.get("gender")
            ),
            "nationality": self.validator.is_valid_nationality(
                merged_data.get("nationality")
            ),
        }
        readable_fields = [
            field_name
            for field_name, is_valid in field_checks.items()
            if is_valid
        ]

        has_low_image_score = bool(
            cropped_quality.get("is_blurry")
            or cropped_quality.get("is_too_dark")
        )
        minimum_fields = (
            self.MINIMUM_READABLE_CORE_FIELDS
            if has_low_image_score
            else 2
        )
        meaningful_lines = sum(
            bool(re.search(r"[A-Za-zÀ-ỹ0-9]{3}", str(line)))
            for line in raw_text
        )
        is_readable = bool(
            len(readable_fields) >= minimum_fields
            and meaningful_lines >= 3
        )

        warnings: list[str] = []
        if cropped_quality.get("is_blurry"):
            warnings.append(
                "Ảnh hơi mờ; kết quả đã được giữ vì OCR vẫn đọc được "
                "các trường cốt lõi."
            )
        if cropped_quality.get("is_too_dark"):
            warnings.append(
                "Vùng CCCD thiếu sáng; nên đối chiếu lại các trường có "
                "độ tin cậy thấp."
            )
        missing_fields = [
            field_name
            for field_name, is_valid in field_checks.items()
            if not is_valid
        ]
        if missing_fields:
            warnings.append(
                "Chưa xác nhận được trường: " + ", ".join(missing_fields)
            )

        return {
            "isReadable": is_readable,
            "readableCoreFields": readable_fields,
            "missingCoreFields": missing_fields,
            "warnings": warnings,
        }

    @staticmethod
    def calculate_orientation_score(
        ocr_result: dict[str, Any],
    ) -> float:
        """Chấm điểm chiều ảnh từ các nhãn cố định trên mặt trước CCCD."""
        lines = ocr_result.get(
            "normalizedText",
            ocr_result.get("rawText", []),
        )
        if isinstance(lines, str):
            lines = lines.splitlines()
        if not isinstance(lines, list):
            lines = []

        text = remove_accents(
            " ".join(str(line) for line in lines if line)
        ).lower()
        text = re.sub(r"\s+", " ", text)

        weighted_patterns = (
            (r"can\s*cuoc\s*cong\s*dan", 3.0),
            (r"citizen\s*identity\s*card", 2.0),
            (r"ho\s*va\s*ten|full\s*name", 2.0),
            (r"ngay\s*sinh|date\s*of\s*birth", 2.0),
            (r"gioi\s*tinh|\bsex\b", 1.0),
            (r"quoc\s*tich|nationality", 1.0),
            (r"que\s*quan|place\s*of\s*origin", 1.0),
            (r"noi\s*thuong\s*tru|place\s*of\s*residence", 1.0),
            (r"co\s*gia\s*tri\s*den|date\s*of\s*expiry", 1.0),
        )

        score = sum(
            weight
            for pattern, weight in weighted_patterns
            if re.search(pattern, text, flags=re.IGNORECASE)
        )

        if re.search(r"(?<!\d)\d{12}(?!\d)", text):
            score += 3.0
        if re.search(r"\d{1,2}/\d{1,2}/\d{4}", text):
            score += 1.0
        if len(lines) >= 8:
            score += 0.5

        structured_data = ocr_result.get("structuredData", {})
        if isinstance(structured_data, dict):
            identifier = re.sub(
                r"\D",
                "",
                str(structured_data.get("idNumber") or ""),
            )
            if len(identifier) == 12:
                score += 1.0

        return round(float(score), 2)

    def ensure_upright_orientation(
        self,
        card_image: Any,
        enhanced_image: Any,
        card_output_path: Path,
        enhanced_output_path: Path,
        debug_dir: Path,
        first_ocr_result: dict[str, Any],
    ) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
        """
        Nếu OCR chiều đầu tiên yếu, thử xoay 180 độ và giữ chiều tốt hơn.

        Bước perspective đã đưa ảnh dọc về ngang. Việc thử 180 độ này xử
        lý cả ảnh úp ngược lẫn trường hợp xoay 90 độ theo hướng còn lại.
        """
        first_score = self.calculate_orientation_score(first_ocr_result)
        orientation_info: dict[str, Any] = {
            "contentRotationDegrees": 0,
            "orientationRetried": False,
            "initialScore": first_score,
            "rotatedScore": None,
        }

        final_debug_path = debug_dir / "detector_05_oriented.jpg"

        if first_score >= self.ORIENTATION_RETRY_THRESHOLD:
            cv2.imwrite(str(final_debug_path), card_image)
            orientation_info["selectedScore"] = first_score
            return (
                card_image,
                enhanced_image,
                first_ocr_result,
                orientation_info,
            )

        rotated_card = cv2.rotate(card_image, cv2.ROTATE_180)
        rotated_enhanced = cv2.rotate(enhanced_image, cv2.ROTATE_180)
        orientation_info["orientationRetried"] = True

        candidate_saved = cv2.imwrite(
            str(card_output_path),
            rotated_card,
        )
        enhanced_candidate_saved = cv2.imwrite(
            str(enhanced_output_path),
            rotated_enhanced,
        )

        if not candidate_saved or not enhanced_candidate_saved:
            cv2.imwrite(str(card_output_path), card_image)
            cv2.imwrite(str(enhanced_output_path), enhanced_image)
            cv2.imwrite(str(final_debug_path), card_image)
            orientation_info["retryError"] = (
                "Không thể lưu ảnh ứng viên xoay 180 độ"
            )
            orientation_info["selectedScore"] = first_score
            return (
                card_image,
                enhanced_image,
                first_ocr_result,
                orientation_info,
            )

        rotated_ocr_result = self.run_full_card_ocr(card_output_path)
        rotated_score = self.calculate_orientation_score(rotated_ocr_result)
        orientation_info["rotatedScore"] = rotated_score

        if rotated_score >= first_score + self.ORIENTATION_SELECTION_MARGIN:
            cv2.imwrite(str(final_debug_path), rotated_card)
            orientation_info["contentRotationDegrees"] = 180
            orientation_info["selectedScore"] = rotated_score
            return (
                rotated_card,
                rotated_enhanced,
                rotated_ocr_result,
                orientation_info,
            )

        # Chiều ban đầu tốt hơn hoặc hai chiều chưa đủ chênh lệch.
        cv2.imwrite(str(card_output_path), card_image)
        cv2.imwrite(str(enhanced_output_path), enhanced_image)
        cv2.imwrite(str(final_debug_path), card_image)
        orientation_info["selectedScore"] = first_score
        return (
            card_image,
            enhanced_image,
            first_ocr_result,
            orientation_info,
        )

    def run_field_ocr(
        self,
        card_image_path: Path,
        field_output_dir: Path,
        layout_y_offset: float = 0.0,
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
                    layout_y_offset=layout_y_offset,
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

    def save_json_response(
        self,
        response: dict[str, Any],
        json_path: Path,
    ) -> None:
        """
        Lưu toàn bộ response OCR vào tệp JSON UTF-8.
        """

        try:
            json_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            safe_response = self.make_json_safe(
                response
            )

            with json_path.open(
                mode="w",
                encoding="utf-8",
            ) as json_file:
                json.dump(
                    safe_response,
                    json_file,
                    ensure_ascii=False,
                    indent=4,
                )

            print(
                f"[INFO] Đã lưu JSON: "
                f"{json_path}"
            )

        except Exception as error:
            print(
                "[WARNING] Không thể lưu JSON "
                f"{json_path}: {error}"
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
        image_file: Path | None = None,
        rejection: dict[str, Any] | None = None,
        partial_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Tạo response khi pipeline thất bại và cố gắng lưu JSON lỗi.
        """

        processing_time = round(
            time.perf_counter()
            - start_time,
            3,
        )

        partial = partial_result or {}
        partial_text_boxes = self.make_json_safe(
            partial.get("textBoxes", [])
        )
        partial_cccd_data = partial.get("cccdData")
        if not isinstance(partial_cccd_data, dict):
            partial_cccd_data = {
                field_name: None
                for field_name in self.FIELD_NAMES
            }

        response = {
            "status": "OCR_FAILED",
            "message": message,
            "cccdData": self.make_json_safe(partial_cccd_data),
            "metadata": {
                "engine": "EasyOCR",
                "processingTime": (
                    processing_time
                ),
                "averageConfidence": self.calculate_average_confidence(
                    partial_text_boxes
                ),
                "fieldConfidences": self.make_json_safe(
                    partial.get("fieldConfidences", {})
                ),
                "validation": self.make_json_safe(
                    partial.get("validation")
                    or {
                        "isValid": False,
                        "errors": [message],
                    }
                ),
            },
            "portrait": None,
            "rawText": self.make_json_safe(partial.get("rawText", [])),
            "textBoxes": partial_text_boxes,
            "mergedTextBoxes": self.make_json_safe(
                partial.get("mergedTextBoxes", [])
            ),
            "fieldResults": self.make_json_safe(
                partial.get("fieldResults", {})
            ),
        }

        if rejection:
            response["metadata"]["rejection"] = self.make_json_safe(
                rejection
            )

        if image_file is not None:
            json_output_dir = (
                Path("storage")
                / "json"
            )

            json_output_path = (
                json_output_dir
                / f"{image_file.stem}_error.json"
            )

            response["metadata"]["inputImage"] = (
                str(image_file)
            )
            response["metadata"]["jsonOutput"] = (
                str(json_output_path)
            )

            self.save_json_response(
                response=response,
                json_path=json_output_path,
            )

        return response

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
