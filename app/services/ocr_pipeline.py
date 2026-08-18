from __future__ import annotations

import json
import re
import time
from collections.abc import Collection
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.config import settings
from app.core.exceptions import BadRequestException
from app.modules.card_detection.detector import CardDetector
from app.modules.card_detection.geometry_refiner import GeometryRefiner
from app.modules.card_detection.image_enhancer import cccd_image_enhancer
from app.modules.ocr.card_side_classifier import classify_cccd_side
from app.modules.ocr.field_ocr_service import field_ocr_service
from app.modules.ocr.line_merger import OCRLineMerger
from app.modules.ocr.result_fuser import (
    estimate_address_crop_layout,
    estimate_field_crop_layout,
    estimate_layout_y_offset,
    fuse_ocr_data,
    remove_accents,
)
from app.modules.ocr.service import ocr_service
from app.modules.ocr.validator import CCCDValidator
from app.modules.qr.cccd_qr_decoder import CCCDQRDecoder
from app.modules.qr.qr_ocr_fuser import (
    build_qr_reference_data,
    fuse_qr_data,
    select_qr_field_ocr_skips,
)
from app.modules.qr.qr_debug import (
    save_parser_qr_overlay,
    save_qr_debug_images,
)
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
        -> QR Fast Path
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
    GEOMETRY_RETRY_SCORE = 12.0
    GEOMETRY_SELECTION_MARGIN = 0.60
    SKEW_SELECTION_MARGIN = 0.25

    def __init__(self) -> None:
        self.card_detector = CardDetector()
        self.geometry_refiner = GeometryRefiner()
        self.ocr_service = ocr_service
        self.field_ocr_service = field_ocr_service
        if getattr(self.field_ocr_service, "engine", None) is None:
            self.field_ocr_service.engine = self.ocr_service.engine
        self.validator = CCCDValidator()
        self.qr_decoder = CCCDQRDecoder(
            time_budget_ms=settings.qr_decode_budget_ms,
        )

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
        stage_timings_ms: dict[str, float] = {}
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

        detection_started = time.perf_counter()
        try:
            detection_result = (
                self.card_detector.detect_from_path(
                    image_path=str(image_file),
                    output_dir=str(debug_dir),
                )
            )
        except BadRequestException as error:
            stage_timings_ms["cardDetection"] = round(
                (time.perf_counter() - detection_started) * 1000.0,
                2,
            )
            rejection = dict(error.data or {})
            rejection.setdefault("errorCode", "CARD_DETECTION_FAILED")
            rejection.setdefault("stage", "CARD_DETECTION")
            rejection.setdefault("reason", error.message)
            return self.build_error_response(
                message=error.message,
                start_time=start_time,
                image_file=image_file,
                rejection=rejection,
                partial_result={
                    "debugDir": str(debug_dir),
                    "processingStagesMs": stage_timings_ms,
                },
            )
        except Exception as error:
            stage_timings_ms["cardDetection"] = round(
                (time.perf_counter() - detection_started) * 1000.0,
                2,
            )
            return self.build_error_response(
                message=(
                    "Phát hiện vùng CCCD thất bại: "
                    f"{error}"
                ),
                start_time=start_time,
                image_file=image_file,
                rejection={
                    "errorCode": "CARD_DETECTION_FAILED",
                    "stage": "CARD_DETECTION",
                    "reason": "Không phát hiện được vùng CCCD",
                    "suggestion": (
                        "Vui lòng đặt trọn một CCCD trong khung hình."
                    ),
                },
                partial_result={
                    "debugDir": str(debug_dir),
                    "processingStagesMs": stage_timings_ms,
                },
            )

        stage_timings_ms["cardDetection"] = round(
            (time.perf_counter() - detection_started) * 1000.0,
            2,
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

        # Quét QR trước OCR. Ngoài việc cấp dữ liệu chuẩn, vị trí QR còn cho
        # biết mặt trước đang đúng chiều (góc trên-phải) hay xoay 180 độ
        # (góc dưới-trái), nhờ vậy tránh một lượt EasyOCR chỉ để thử chiều.
        qr_probe_started = time.perf_counter()
        qr_result = self.decode_qr_fast_path(card_image)
        qr_probe_attempt_count = int(
            qr_result.get("attemptCount", 0) or 0
        )
        qr_probe_elapsed_ms = float(qr_result.get("elapsedMs", 0.0) or 0.0)
        qr_rotation_degrees = self.qr_orientation_rotation(
            qr_result,
            card_size=(int(card_image.shape[1]), int(card_image.shape[0])),
        )
        qr_orientation_applied = qr_rotation_degrees == 0
        if qr_rotation_degrees == 180:
            original_card_image = card_image
            original_enhanced_image = enhanced_image
            rotated_card_image = cv2.rotate(card_image, cv2.ROTATE_180)
            rotated_enhanced_image = cv2.rotate(
                enhanced_image,
                cv2.ROTATE_180,
            )
            card_saved = cv2.imwrite(
                str(card_output_path),
                rotated_card_image,
            )
            enhanced_saved = cv2.imwrite(
                str(enhanced_output_path),
                rotated_enhanced_image,
            )
            if card_saved and enhanced_saved:
                card_image = rotated_card_image
                enhanced_image = rotated_enhanced_image
                qr_orientation_applied = True
                rotated_qr_result = self.decode_qr_fast_path(card_image)
                rotated_qr_result["attemptCount"] = (
                    qr_probe_attempt_count
                    + int(rotated_qr_result.get("attemptCount", 0) or 0)
                )
                rotated_qr_result["elapsedMs"] = round(
                    qr_probe_elapsed_ms
                    + float(rotated_qr_result.get("elapsedMs", 0.0) or 0.0),
                    2,
                )
                qr_result = rotated_qr_result
            else:
                cv2.imwrite(str(card_output_path), original_card_image)
                cv2.imwrite(
                    str(enhanced_output_path),
                    original_enhanced_image,
                )
                qr_result.setdefault("errors", []).append(
                    "QR_ORIENTATION_SAVE_FAILED"
                )
        qr_result["orientationRotationDegrees"] = qr_rotation_degrees
        qr_result["orientationProbeAttemptCount"] = qr_probe_attempt_count
        qr_result["orientationProbeElapsedMs"] = qr_probe_elapsed_ms
        stage_timings_ms["qrOrientationProbe"] = round(
            (time.perf_counter() - qr_probe_started) * 1000.0,
            2,
        )

        # OCR chính chạy trên ảnh warped gốc đã được QR chuẩn hóa chiều.
        initial_ocr_started = time.perf_counter()
        full_ocr_result = self.run_full_card_ocr(
            image_path=card_output_path,
        )
        stage_timings_ms["initialFullCardOcr"] = round(
            (time.perf_counter() - initial_ocr_started) * 1000.0,
            2,
        )

        # Chuẩn hóa chiều ảnh chính trước. Nếu OCR hình học chạy khi thẻ đang
        # úp ngược, mọi ứng viên đều có điểm thấp và pipeline phải OCR lại ba
        # ảnh không cần thiết trước khi mới thử xoay 180 độ.
        orientation_started = time.perf_counter()
        if qr_result.get("decoded") and qr_orientation_applied:
            orientation_info = {
                "contentRotationDegrees": qr_rotation_degrees,
                "orientationRetried": False,
                "orientationSource": "CCCD_QR_POSITION",
                "initialScore": self.calculate_orientation_score(
                    full_ocr_result
                ),
                "selectedScore": self.calculate_orientation_score(
                    full_ocr_result
                ),
            }
            cv2.imwrite(
                str(debug_dir / "detector_05_oriented.jpg"),
                card_image,
            )
        else:
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
        stage_timings_ms["orientation"] = round(
            (time.perf_counter() - orientation_started) * 1000.0,
            2,
        )

        geometry_selection_started = time.perf_counter()
        (
            card_image,
            enhanced_image,
            full_ocr_result,
            geometry_selection,
        ) = self.select_best_geometry_candidate(
            card_image=card_image,
            enhanced_image=enhanced_image,
            first_ocr_result=full_ocr_result,
            detection_result=detection_result,
            card_output_path=card_output_path,
            enhanced_output_path=enhanced_output_path,
            debug_dir=debug_dir,
            content_rotation_degrees=int(
                orientation_info.get("contentRotationDegrees", 0)
            ),
        )
        stage_timings_ms["geometrySelection"] = round(
            (time.perf_counter() - geometry_selection_started) * 1000.0,
            2,
        )

        skew_started = time.perf_counter()
        (
            card_image,
            enhanced_image,
            full_ocr_result,
            skew_info,
        ) = self.refine_residual_skew(
            card_image=card_image,
            enhanced_image=enhanced_image,
            first_ocr_result=full_ocr_result,
            card_output_path=card_output_path,
            enhanced_output_path=enhanced_output_path,
            debug_dir=debug_dir,
            qr_decoded=bool(qr_result.get("decoded")),
        )
        stage_timings_ms["residualSkew"] = round(
            (time.perf_counter() - skew_started) * 1000.0,
            2,
        )
        orientation_info["geometrySelection"] = geometry_selection
        orientation_info["residualSkew"] = skew_info
        orientation_info["selectedScore"] = (
            self.calculate_orientation_score(full_ocr_result)
        )
        selected_geometry = detection_result.get("geometry", {})
        if isinstance(selected_geometry, dict):
            selected_geometry["residualSkewCorrection"] = skew_info
        detection_result["cardImage"] = card_image
        detection_result["enhancedImage"] = enhanced_image

        # Chấm chất lượng trên đúng ảnh cuối cùng đã được chọn. Laplacian
        # thấp chỉ là cảnh báo; quyết định từ chối vẫn dựa thêm vào OCR.
        blur_info = cccd_image_enhancer.estimate_blur(card_image)
        cropped_quality = check_image_quality(
            card_image,
            blur_threshold=self.CROP_BLUR_WARNING_THRESHOLD,
            dark_threshold=self.CROP_DARK_WARNING_THRESHOLD,
        )

        # Ảnh warped gốc luôn là nguồn OCR chính. Chỉ thử ảnh enhanced cũ
        # khi ảnh mờ/tối hoặc OCR chiều ảnh yếu; kết quả enhanced phải hơn
        # ảnh gốc đủ biên mới được dùng. Không dùng adaptive variant để thay
        # toàn bộ text box vì thay đổi nhỏ về confidence có thể làm Layout
        # Parser mất nhãn và kéo theo mất nhiều trường.
        enhancement_retry_started = time.perf_counter()
        full_ocr_variant = "card"
        enhanced_ocr_retried = False
        if (
            cropped_quality["is_blurry"]
            or cropped_quality["is_too_dark"]
            or float(orientation_info.get("selectedScore", 0.0))
            < self.ORIENTATION_RETRY_THRESHOLD + 3.0
        ):
            enhanced_ocr_retried = True
            enhanced_ocr_result = self.run_full_card_ocr(
                image_path=enhanced_output_path,
            )
            full_ocr_result, full_ocr_variant = (
                self.select_better_full_ocr_result(
                    card_result=full_ocr_result,
                    enhanced_result=enhanced_ocr_result,
                )
            )
        stage_timings_ms["qualityAndEnhancedRetry"] = round(
            (time.perf_counter() - enhancement_retry_started) * 1000.0,
            2,
        )

        # Nếu probe ban đầu chưa giải mã được, thử đúng một lần trên ảnh cuối
        # sau orientation/geometry. Kết quả QR đã hợp lệ được tái sử dụng,
        # tránh quét lặp lại cùng ảnh ở cuối pipeline.
        qr_retry_started = time.perf_counter()
        if settings.qr_fast_path_enabled and not qr_result.get("decoded"):
            initial_attempt_count = int(
                qr_result.get("attemptCount", 0) or 0
            )
            initial_elapsed_ms = float(qr_result.get("elapsedMs", 0.0) or 0.0)
            initial_errors = list(qr_result.get("errors", []) or [])
            final_qr_result = self.decode_qr_fast_path(card_image)
            final_qr_result["attemptCount"] = (
                initial_attempt_count
                + int(final_qr_result.get("attemptCount", 0) or 0)
            )
            final_qr_result["elapsedMs"] = round(
                initial_elapsed_ms
                + float(final_qr_result.get("elapsedMs", 0.0) or 0.0),
                2,
            )
            final_qr_result["orientationProbeAttemptCount"] = (
                qr_probe_attempt_count
            )
            final_qr_result["orientationProbeElapsedMs"] = (
                qr_probe_elapsed_ms
            )
            final_qr_result["orientationProbeErrors"] = initial_errors
            final_qr_result["orientationRotationDegrees"] = int(
                orientation_info.get("contentRotationDegrees", 0) or 0
            )
            qr_result = final_qr_result
        stage_timings_ms["qrFastPath"] = round(
            stage_timings_ms.get("qrOrientationProbe", 0.0)
            + (time.perf_counter() - qr_retry_started) * 1000.0,
            2,
        )
        qr_debug = save_qr_debug_images(
            card_image=card_image,
            qr_result=qr_result,
            debug_dir=debug_dir,
        )
        qr_result["debug"] = qr_debug
        qr_data = self.make_json_safe(
            qr_result.get("structuredData", {})
        )
        if not isinstance(qr_data, dict):
            qr_data = {}

        selected_full_card_data = self.make_json_safe(
            full_ocr_result.get("structuredData", {})
        )
        if not isinstance(selected_full_card_data, dict):
            selected_full_card_data = {}

        side_started = time.perf_counter()
        card_side = classify_cccd_side(full_ocr_result)
        stage_timings_ms["cardSideClassification"] = round(
            (time.perf_counter() - side_started) * 1000.0,
            2,
        )
        orientation_info["cardSide"] = card_side
        if card_side.get("side") == "BACK":
            processing_time = round(
                time.perf_counter() - start_time,
                3,
            )
            stage_timings_ms["total"] = round(
                processing_time * 1000.0,
                2,
            )
            qr_public = self.public_qr_metadata(
                qr_result=qr_result,
                qr_fusion={},
                skipped_fields=set(),
            )
            side_validation = {
                "isValid": False,
                "errors": [
                    "Ảnh là mặt sau CCCD; pipeline hiện cần mặt trước."
                ],
            }
            return self.build_error_response(
                message=(
                    "Đã nhận diện mặt sau CCCD; cần ảnh mặt trước để đọc "
                    "đủ thông tin và lấy ảnh chân dung"
                ),
                start_time=start_time,
                image_file=image_file,
                rejection={
                    "errorCode": "CCCD_BACK_SIDE_DETECTED",
                    "stage": "CARD_SIDE_CLASSIFICATION",
                    "reason": "Ảnh đầu vào là mặt sau CCCD",
                    "detectedSide": "BACK",
                    "suggestion": (
                        "Vui lòng chụp mặt trước có số định danh, họ tên "
                        "và ảnh chân dung."
                    ),
                    "debugImages": qr_debug,
                },
                partial_result={
                    "cccdData": selected_full_card_data,
                    "validation": side_validation,
                    "rawText": full_ocr_result.get("normalizedText", []),
                    "textBoxes": full_ocr_result.get("textBoxes", []),
                    "mergedTextBoxes": full_ocr_result.get(
                        "mergedTextBoxes",
                        [],
                    ),
                    "qrFastPath": qr_public,
                    "imageQuality": self.make_json_safe({
                        **blur_info,
                        "cropBlurScore": cropped_quality["blur_score"],
                        "brightnessScore": cropped_quality[
                            "brightness_score"
                        ],
                        "decision": "REJECTED_WRONG_SIDE",
                        "warnings": [],
                        "readableCoreFields": [],
                    }),
                    "cardSide": card_side,
                    "geometry": detection_result.get("geometry", {}),
                    "orientation": orientation_info,
                    "processingStagesMs": stage_timings_ms,
                    "debugDir": str(debug_dir),
                    "cardImage": str(card_output_path),
                    "enhancedImage": str(enhanced_output_path),
                    "reviewRequired": True,
                    "parserDiagnostics": {
                        "status": "WRONG_CARD_SIDE",
                        "validFields": [],
                        "missingFields": list(self.FIELD_NAMES),
                        "cardSide": card_side,
                        "errors": side_validation["errors"],
                    },
                },
            )

        raw_text_for_fusion = self.make_json_safe(
            full_ocr_result.get(
                "normalizedText",
                full_ocr_result.get("rawText", []),
            )
        )
        if isinstance(raw_text_for_fusion, str):
            raw_text_for_fusion = raw_text_for_fusion.splitlines()
        if not isinstance(raw_text_for_fusion, list):
            raw_text_for_fusion = []

        # Hợp nhất sơ bộ trước field OCR để lấy reference từ structuredData,
        # raw text và vị trí text box. Khi lần OCR field đầu tiên khớp reference
        # hợp lệ, FieldOCRService có thể dừng sớm thay vì thử 3-6 biến thể.
        preliminary_card_data, _ = fuse_ocr_data(
            full_card_data=selected_full_card_data,
            field_data={},
            raw_text=raw_text_for_fusion,
            field_results={},
            text_boxes=full_ocr_result.get("textBoxes", []),
            image_size=(
                int(card_image.shape[1]),
                int(card_image.shape[0]),
            ),
        )

        qr_field_ocr_skips: set[str] = set()
        if (
            settings.qr_skip_confirmed_field_ocr
            and qr_result.get("decoded")
        ):
            qr_field_ocr_skips = select_qr_field_ocr_skips(
                qr_data=qr_data,
                full_card_data=preliminary_card_data,
                validator=self.validator,
            )
        field_reference_data = build_qr_reference_data(
            full_card_data=preliminary_card_data,
            qr_data=qr_data,
            validator=self.validator,
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
        address_layout = estimate_address_crop_layout(
            full_ocr_result.get("textBoxes", []),
            image_size=(
                int(card_image.shape[1]),
                int(card_image.shape[0]),
            ),
        )
        orientation_info["addressCropLayout"] = address_layout
        field_layout = estimate_field_crop_layout(
            full_ocr_result.get("textBoxes", []),
            image_size=(
                int(card_image.shape[1]),
                int(card_image.shape[0]),
            ),
        )
        field_layout = self.attach_qr_region_to_field_layout(
            field_layout=field_layout,
            qr_result=qr_result,
            card_size=(
                int(card_image.shape[1]),
                int(card_image.shape[0]),
            ),
        )
        orientation_info["fieldCropLayout"] = field_layout

        field_ocr_started = time.perf_counter()
        field_ocr_result = self.run_field_ocr(
            card_image_path=card_output_path,
            field_output_dir=field_output_dir,
            layout_y_offset=layout_y_offset,
            address_layout=address_layout,
            field_layout=field_layout,
            reference_data=field_reference_data,
            skip_fields=qr_field_ocr_skips,
        )
        parser_qr_overlay = save_parser_qr_overlay(
            fields_debug_path=field_output_dir / "fields_debug.jpg",
            qr_result=qr_result,
            card_size=(
                int(card_image.shape[1]),
                int(card_image.shape[0]),
            ),
            output_dir=field_output_dir,
        )
        value_parser_qr_overlay = save_parser_qr_overlay(
            fields_debug_path=field_output_dir / "fields_value_debug.jpg",
            qr_result=qr_result,
            card_size=(
                int(card_image.shape[1]),
                int(card_image.shape[0]),
            ),
            output_dir=field_output_dir,
            output_name="fields_values_parser_qr_debug.jpg",
        )
        field_debug_result = field_ocr_result.setdefault("debug", {})
        if parser_qr_overlay:
            qr_debug["parserOverlay"] = parser_qr_overlay
            qr_result["debug"] = qr_debug
            if isinstance(field_debug_result, dict):
                field_debug_result["parserQrOverlay"] = parser_qr_overlay
        if value_parser_qr_overlay:
            qr_debug["parserValueOverlay"] = value_parser_qr_overlay
            qr_result["debug"] = qr_debug
            if isinstance(field_debug_result, dict):
                field_debug_result["parserValueQrOverlay"] = (
                    value_parser_qr_overlay
                )
        stage_timings_ms["fieldCropAndOcr"] = round(
            (time.perf_counter() - field_ocr_started) * 1000.0,
            2,
        )

        full_card_data = selected_full_card_data

        field_data = self.make_json_safe(
            field_ocr_result.get(
                "structuredData",
                {},
            )
        )

        fusion_started = time.perf_counter()
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

        merged_data, data_sources, qr_fusion = fuse_qr_data(
            ocr_data=merged_data,
            ocr_sources=data_sources,
            qr_data=qr_data,
            validator=self.validator,
        )

        validation_result = self.validator.validate(
            merged_data
        )
        qr_conflict_fields = [
            str(item.get("field"))
            for item in qr_fusion.get("conflicts", [])
            if (
                isinstance(item, dict)
                and item.get("field")
                and bool(item.get("requiresReview", True))
            )
        ]
        qr_advisory_fields = [
            str(item.get("field"))
            for item in qr_fusion.get("conflicts", [])
            if (
                isinstance(item, dict)
                and item.get("field")
                and not bool(item.get("requiresReview", True))
            )
        ]
        if qr_conflict_fields:
            validation_result["isValid"] = False
            validation_result.setdefault("errors", []).append(
                "Thông tin QR không khớp chữ in trên thẻ: "
                + ", ".join(qr_conflict_fields)
            )
            validation_result["qrConflictFields"] = qr_conflict_fields
        if qr_advisory_fields:
            validation_result["qrAdvisoryDifferenceFields"] = (
                qr_advisory_fields
            )
        stage_timings_ms["fusionAndValidation"] = round(
            (time.perf_counter() - fusion_started) * 1000.0,
            2,
        )

        field_confidences = self.get_field_confidences(
            field_ocr_result
        )
        qr_public_metadata = self.public_qr_metadata(
            qr_result=qr_result,
            qr_fusion=qr_fusion,
            skipped_fields=qr_field_ocr_skips,
        )
        parser_diagnostics = self.build_parser_diagnostics(
            merged_data=merged_data,
            data_sources=data_sources,
            field_confidences=field_confidences,
            validation_result=validation_result,
            card_side=card_side,
            qr_metadata=qr_public_metadata,
        )

        readability = self.evaluate_ocr_readability(
            merged_data=merged_data,
            cropped_quality=cropped_quality,
            raw_text=raw_text_for_fusion,
        )

        if not readability["isReadable"]:
            low_quality = bool(
                cropped_quality.get("is_blurry")
                or cropped_quality.get("is_too_dark")
            )
            error_code = (
                "OCR_CORE_FIELDS_MISSING_LOW_QUALITY"
                if low_quality
                else "OCR_CORE_FIELDS_MISSING"
            )
            reason = (
                "Chất lượng vùng thẻ thấp và parser không xác nhận được "
                "đủ trường cốt lõi"
                if low_quality
                else (
                    "Ảnh đủ nét/sáng nhưng parser chưa xác nhận được đủ "
                    "trường; cần kiểm tra lại vùng cắt hoặc mặt thẻ"
                )
            )
            suggestion = (
                "Giữ máy ổn định, lấy nét vào chữ và tăng ánh sáng rồi "
                "chụp lại."
                if low_quality
                else (
                    "Mở detector_03_geometry_selected.jpg và "
                    "fields/fields_parser_qr_debug.jpg để kiểm tra vùng "
                    "cắt trước khi thay đổi ngưỡng độ nét."
                )
            )
            processing_time = round(
                time.perf_counter() - start_time,
                3,
            )
            stage_timings_ms["total"] = round(
                processing_time * 1000.0,
                2,
            )
            image_quality_metadata = self.make_json_safe({
                **blur_info,
                "cropBlurScore": cropped_quality["blur_score"],
                "brightnessScore": cropped_quality["brightness_score"],
                "decision": (
                    "FAILED_LOW_QUALITY"
                    if low_quality
                    else "PASSED_IMAGE_FAILED_PARSER"
                ),
                "warnings": readability["warnings"],
                "readableCoreFields": readability[
                    "readableCoreFields"
                ],
                "missingCoreFields": readability[
                    "missingCoreFields"
                ],
            })
            return self.build_error_response(
                message=(
                    "Parser chưa xác nhận được đủ thông tin cốt lõi từ CCCD"
                ),
                start_time=start_time,
                image_file=image_file,
                rejection={
                    "errorCode": error_code,
                    "stage": "PARSER_VALIDATION",
                    "reason": reason,
                    "blurScore": cropped_quality["blur_score"],
                    "brightnessScore": (
                        cropped_quality["brightness_score"]
                    ),
                    "readableCoreFields": readability[
                        "readableCoreFields"
                    ],
                    "missingCoreFields": readability[
                        "missingCoreFields"
                    ],
                    "suggestion": suggestion,
                    "debugImages": {
                        "qr": qr_debug.get("detectionImage"),
                        "parser": qr_debug.get("parserOverlay"),
                        "fields": str(field_output_dir / "fields_debug.jpg"),
                    },
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
                    "fieldConfidences": field_confidences,
                    "dataSources": data_sources,
                    "qrFastPath": qr_public_metadata,
                    "parserDiagnostics": parser_diagnostics,
                    "imageQuality": image_quality_metadata,
                    "cardSide": card_side,
                    "geometry": detection_result.get("geometry", {}),
                    "orientation": orientation_info,
                    "processingStagesMs": stage_timings_ms,
                    "debugDir": str(debug_dir),
                    "cardImage": str(card_output_path),
                    "enhancedImage": str(enhanced_output_path),
                    "reviewRequired": True,
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
        stage_timings_ms["total"] = round(processing_time * 1000.0, 2)

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
        field_ocr_attempt_count = sum(
            int(item.get("attemptCount", 0) or 0)
            for item in field_results.values()
            if isinstance(item, dict)
        )
        full_card_ocr_attempt_count = (
            1
            + int(bool(orientation_info.get("orientationRetried")))
            + max(
                0,
                len(geometry_selection.get("candidates", [])) - 1,
            )
            + len(skew_info.get("candidates", []))
            + int(enhanced_ocr_retried)
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
                "engine": (
                    "EasyOCR + ZXing-C++/OpenCV QR"
                    if qr_result.get("decoded")
                    else "EasyOCR"
                ),
                "processingTime": processing_time,
                "processingStagesMs": stage_timings_ms,
                "fullCardOcrAttemptCount": full_card_ocr_attempt_count,
                "fieldOcrAttemptCount": field_ocr_attempt_count,
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
                        "PASSED_WITH_WARNING"
                        if not cropped_quality["is_valid"]
                        else "PASSED"
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
                "parserDiagnostics": self.make_json_safe(
                    parser_diagnostics
                ),
                "cardSide": self.make_json_safe(card_side),
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
                "qrFastPath": qr_public_metadata,
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

    def decode_qr_fast_path(self, card_image: Any) -> dict[str, Any]:
        """Chạy QR an toàn; lỗi thư viện không được làm hỏng OCR fallback."""
        started = time.perf_counter()
        if settings.qr_fast_path_enabled:
            try:
                return self.qr_decoder.decode(card_image)
            except Exception as error:
                result = self.qr_decoder.empty_result()
                result["errors"] = [
                    f"QR_FAST_PATH_ERROR:{type(error).__name__}"
                ]
                return self.qr_decoder.finalize_result(result, started)

        result = self.qr_decoder.empty_result()
        result["errors"] = ["QR_FAST_PATH_DISABLED"]
        return self.qr_decoder.finalize_result(result, started)

    @staticmethod
    def qr_orientation_rotation(
        qr_result: dict[str, Any],
        card_size: tuple[int, int],
    ) -> int:
        """QR dưới-trái của mặt trước cho biết ảnh đang bị xoay 180 độ."""
        if not qr_result.get("decoded"):
            return 0
        card_width = max(int(card_size[0]), 1)
        card_height = max(int(card_size[1]), 1)
        box = qr_result.get("boundingBox")
        if isinstance(box, dict):
            center_x = float(box.get("x", 0)) + float(box.get("width", 0)) / 2
            center_y = float(box.get("y", 0)) + float(box.get("height", 0)) / 2
        else:
            try:
                points = np.asarray(
                    qr_result.get("polygon", []),
                    dtype=np.float32,
                ).reshape(-1, 2)
                if len(points) < 4:
                    return 0
                points = cv2.convexHull(points).reshape(-1, 2)
            except (AttributeError, TypeError, ValueError, cv2.error):
                return 0
            center_x = float(points[:, 0].mean())
            center_y = float(points[:, 1].mean())
        return 180 if (
            center_x < card_width * 0.50
            and center_y > card_height * 0.50
        ) else 0

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

    def score_geometry_ocr_result(
        self,
        ocr_result: dict[str, Any],
    ) -> float:
        """Chấm OCR khi so sánh các cách nắn, có tính lượng chữ giữ lại."""
        score = self.score_full_ocr_result(ocr_result)
        text_boxes = ocr_result.get("textBoxes", [])
        if not isinstance(text_boxes, list):
            text_boxes = []

        meaningful_boxes = 0
        meaningful_characters = 0
        for item in text_boxes:
            if isinstance(item, dict):
                text = str(item.get("text") or item.get("originalText") or "")
                confidence = float(item.get("confidence", 0.0) or 0.0)
            else:
                text = str(getattr(item, "text", "") or "")
                confidence = float(getattr(item, "confidence", 0.0) or 0.0)
            characters = re.findall(r"[A-Za-zÀ-ỹ0-9]", text)
            if confidence < 0.08 or len(characters) < 2:
                continue
            meaningful_boxes += 1
            meaningful_characters += len(characters)

        score += min(1.5, meaningful_boxes * 0.075)
        score += min(1.5, meaningful_characters * 0.006)
        return round(float(score), 3)

    def select_best_geometry_candidate(
        self,
        card_image: Any,
        enhanced_image: Any,
        first_ocr_result: dict[str, Any],
        detection_result: dict[str, Any],
        card_output_path: Path,
        enhanced_output_path: Path,
        debug_dir: Path,
        content_rotation_degrees: int = 0,
    ) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
        """Thử các quad an toàn và chỉ nhận ứng viên có OCR tốt hơn."""
        primary_geometry = detection_result.get("geometry", {})
        if not isinstance(primary_geometry, dict):
            primary_geometry = {}
        primary_name = str(
            primary_geometry.get("candidateName", "perspective_contour")
        )
        primary_score = self.score_geometry_ocr_result(first_ocr_result)
        candidates = detection_result.get("cardCandidates", [])
        if not isinstance(candidates, list):
            candidates = []

        source_coverage = float(
            primary_geometry.get("sourceCoverageRatio", 0.0) or 0.0
        )
        perspective_severity = float(
            primary_geometry.get("perspectiveSeverity", 0.0) or 0.0
        )
        whole_card_reliable = bool(
            primary_geometry.get("wholeCardReliable")
        )
        first_candidate = candidates[0] if candidates else {}
        first_candidate_geometry = (
            first_candidate.get("geometry", {})
            if isinstance(first_candidate, dict)
            else {}
        )
        first_candidate_metrics = (
            first_candidate_geometry.get("detectionMetrics", {})
            if isinstance(first_candidate_geometry, dict)
            else {}
        )
        has_verified_hough_candidate = bool(
            isinstance(first_candidate_metrics, dict)
            and first_candidate_metrics.get("wholeCardCandidate")
        )
        retry_due_to_geometry = bool(
            not whole_card_reliable
            and (
                has_verified_hough_candidate
                or source_coverage < 0.70
                or perspective_severity >= 0.35
            )
        )
        should_retry = bool(
            candidates
            and retry_due_to_geometry
            and (
                has_verified_hough_candidate
                or primary_score < self.GEOMETRY_RETRY_SCORE
            )
        )
        selection_info: dict[str, Any] = {
            "retried": should_retry,
            "initialCandidate": primary_name,
            "initialScore": primary_score,
            "selectedCandidate": primary_name,
            "selectedScore": primary_score,
            "selectionMargin": self.GEOMETRY_SELECTION_MARGIN,
            "retryReason": (
                "VERIFIED_WHOLE_CARD_CANDIDATE"
                if should_retry and has_verified_hough_candidate
                else "WEAK_OCR_AND_SUSPICIOUS_GEOMETRY"
                if should_retry
                else "PRIMARY_GEOMETRY_ACCEPTED"
            ),
            "maximumOcrCandidates": 1,
            "candidates": [
                {
                    "name": primary_name,
                    "score": primary_score,
                    "selected": True,
                }
            ],
        }

        if not should_retry:
            primary_geometry["selection"] = selection_info
            cv2.imwrite(
                str(debug_dir / "detector_03_geometry_selected.jpg"),
                card_image,
            )
            return (
                card_image,
                enhanced_image,
                first_ocr_result,
                selection_info,
            )

        best_name = primary_name
        best_card = card_image
        best_result = first_ocr_result
        best_score = primary_score
        best_geometry = primary_geometry

        for index, candidate in enumerate(candidates[:1], start=1):
            if not isinstance(candidate, dict):
                continue
            raw_candidate_card = candidate.get("cardImage")
            if raw_candidate_card is None:
                continue
            candidate_card = raw_candidate_card
            if int(content_rotation_degrees) % 360 == 180:
                candidate_card = cv2.rotate(
                    raw_candidate_card,
                    cv2.ROTATE_180,
                )
            candidate_name = str(candidate.get("name", f"candidate_{index}"))
            candidate_path = (
                debug_dir
                / f"detector_03_ocr_candidate_{index:02d}.jpg"
            )
            if not cv2.imwrite(str(candidate_path), candidate_card):
                selection_info["candidates"].append(
                    {
                        "name": candidate_name,
                        "score": None,
                        "error": "Không thể lưu ảnh ứng viên",
                        "selected": False,
                    }
                )
                continue

            candidate_result = self.run_full_card_ocr(candidate_path)
            candidate_score = self.score_geometry_ocr_result(
                candidate_result
            )
            selection_info["candidates"].append(
                {
                    "name": candidate_name,
                    "score": candidate_score,
                    "selected": False,
                }
            )
            if candidate_score > best_score:
                best_name = candidate_name
                best_card = candidate_card
                best_result = candidate_result
                best_score = candidate_score
                geometry = candidate.get("geometry", {})
                best_geometry = geometry if isinstance(geometry, dict) else {}

        if (
            best_name == primary_name
            or best_score < primary_score + self.GEOMETRY_SELECTION_MARGIN
        ):
            primary_geometry["selection"] = selection_info
            cv2.imwrite(
                str(debug_dir / "detector_03_geometry_selected.jpg"),
                card_image,
            )
            return (
                card_image,
                enhanced_image,
                first_ocr_result,
                selection_info,
            )

        try:
            best_enhanced = self.card_detector.enhancer.enhance(
                best_card
            )["final"]
        except (TypeError, ValueError, cv2.error, KeyError) as error:
            selection_info["selectionError"] = (
                f"Không thể tăng cường ứng viên đã chọn: {error}"
            )
            primary_geometry["selection"] = selection_info
            return (
                card_image,
                enhanced_image,
                first_ocr_result,
                selection_info,
            )

        card_saved = cv2.imwrite(str(card_output_path), best_card)
        enhanced_saved = cv2.imwrite(
            str(enhanced_output_path),
            best_enhanced,
        )
        if not card_saved or not enhanced_saved:
            cv2.imwrite(str(card_output_path), card_image)
            cv2.imwrite(str(enhanced_output_path), enhanced_image)
            selection_info["selectionError"] = (
                "Không thể lưu cặp ảnh hình học đã chọn"
            )
            primary_geometry["selection"] = selection_info
            return (
                card_image,
                enhanced_image,
                first_ocr_result,
                selection_info,
            )

        selection_info["selectedCandidate"] = best_name
        selection_info["selectedScore"] = best_score
        for candidate_info in selection_info["candidates"]:
            candidate_info["selected"] = bool(
                candidate_info.get("name") == best_name
            )
        selected_geometry = dict(best_geometry)
        selected_geometry["selection"] = selection_info
        detection_result["geometry"] = selected_geometry
        cv2.imwrite(
            str(debug_dir / "detector_03_geometry_selected.jpg"),
            best_card,
        )
        return (
            best_card,
            best_enhanced,
            best_result,
            selection_info,
        )

    def refine_residual_skew(
        self,
        card_image: Any,
        enhanced_image: Any,
        first_ocr_result: dict[str, Any],
        card_output_path: Path,
        enhanced_output_path: Path,
        debug_dir: Path,
        qr_decoded: bool = False,
    ) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
        """Sửa xiên nhỏ bằng shear và xác nhận lại bằng OCR toàn thẻ."""
        try:
            estimate = self.geometry_refiner.estimate_text_skew(
                card_image,
                first_ocr_result.get("textBoxes", []),
            )
        except (TypeError, ValueError, cv2.error) as error:
            info = {
                "retried": False,
                "selectedCorrectionDegrees": 0.0,
                "error": f"Không thể đo độ xiên: {error}",
            }
            return (
                card_image,
                enhanced_image,
                first_ocr_result,
                info,
            )

        initial_score = self.score_geometry_ocr_result(first_ocr_result)
        correction_angles = self.geometry_refiner.build_correction_angles(
            estimate
        )
        info: dict[str, Any] = {
            "retried": bool(correction_angles),
            "estimate": estimate,
            "initialScore": initial_score,
            "selectedScore": initial_score,
            "selectedCorrectionDegrees": 0.0,
            "correctionMode": "vertical_shear",
            "candidates": [],
        }
        selected_debug_path = debug_dir / "detector_06_deskewed.jpg"
        estimated_angle = abs(float(estimate.get("angleDegrees", 0.0) or 0.0))
        skip_retry_reason: str | None = None
        if (
            correction_angles
            and qr_decoded
            and initial_score >= 18.0
            and estimated_angle <= 3.2
        ):
            skip_retry_reason = "QR_AND_STRONG_OCR"
        elif (
            correction_angles
            and initial_score >= 26.0
            and estimated_angle <= 1.5
        ):
            skip_retry_reason = "STRONG_OCR_SMALL_SKEW"
        if skip_retry_reason:
            info["retried"] = False
            info["retrySkipped"] = True
            info["skipReason"] = skip_retry_reason
            cv2.imwrite(str(selected_debug_path), card_image)
            return (
                card_image,
                enhanced_image,
                first_ocr_result,
                info,
            )
        if not correction_angles:
            cv2.imwrite(str(selected_debug_path), card_image)
            return (
                card_image,
                enhanced_image,
                first_ocr_result,
                info,
            )

        best_card = card_image
        best_enhanced = enhanced_image
        best_result = first_ocr_result
        best_score = initial_score
        best_angle = 0.0
        initial_angle = abs(float(estimate.get("angleDegrees", 0.0)))
        initial_confidence = float(estimate.get("confidence", 0.0))

        for index, correction_angle in enumerate(
            correction_angles,
            start=1,
        ):
            try:
                candidate_card = self.geometry_refiner.correct_vertical_shear(
                    card_image,
                    correction_angle,
                )
                candidate_enhanced = (
                    self.geometry_refiner.correct_vertical_shear(
                        enhanced_image,
                        correction_angle,
                    )
                )
            except (TypeError, ValueError, cv2.error) as error:
                info["candidates"].append(
                    {
                        "correctionDegrees": correction_angle,
                        "score": None,
                        "error": str(error),
                        "selected": False,
                    }
                )
                continue

            candidate_path = (
                debug_dir
                / f"detector_06_skew_candidate_{index:02d}.jpg"
            )
            if not cv2.imwrite(str(candidate_path), candidate_card):
                info["candidates"].append(
                    {
                        "correctionDegrees": correction_angle,
                        "score": None,
                        "error": "Không thể lưu ảnh ứng viên",
                        "selected": False,
                    }
                )
                continue

            candidate_result = self.run_full_card_ocr(candidate_path)
            candidate_score = self.score_geometry_ocr_result(
                candidate_result
            )
            try:
                residual = self.geometry_refiner.estimate_text_skew(
                    candidate_card,
                    candidate_result.get("textBoxes", []),
                )
            except (TypeError, ValueError, cv2.error):
                residual = {
                    "angleDegrees": None,
                    "confidence": 0.0,
                    "reliable": False,
                }

            residual_value = residual.get("angleDegrees")
            residual_angle = (
                abs(float(residual_value))
                if residual_value is not None
                else initial_angle
            )
            candidate_info = {
                "correctionDegrees": correction_angle,
                "score": candidate_score,
                "residualEstimate": residual,
                "selected": False,
            }
            info["candidates"].append(candidate_info)

            score_improved = bool(
                candidate_score >= best_score + self.SKEW_SELECTION_MARGIN
            )
            geometry_improved_without_ocr_loss = bool(
                best_angle == 0.0
                and initial_confidence >= 0.55
                and residual_angle <= initial_angle * 0.55
                and candidate_score >= initial_score - 0.05
            )
            if score_improved or geometry_improved_without_ocr_loss:
                best_card = candidate_card
                best_enhanced = candidate_enhanced
                best_result = candidate_result
                best_score = candidate_score
                best_angle = float(correction_angle)

        if best_angle == 0.0:
            cv2.imwrite(str(selected_debug_path), card_image)
            return (
                card_image,
                enhanced_image,
                first_ocr_result,
                info,
            )

        card_saved = cv2.imwrite(str(card_output_path), best_card)
        enhanced_saved = cv2.imwrite(
            str(enhanced_output_path),
            best_enhanced,
        )
        if not card_saved or not enhanced_saved:
            cv2.imwrite(str(card_output_path), card_image)
            cv2.imwrite(str(enhanced_output_path), enhanced_image)
            info["selectionError"] = (
                "Không thể lưu cặp ảnh sau hiệu chỉnh xiên"
            )
            cv2.imwrite(str(selected_debug_path), card_image)
            return (
                card_image,
                enhanced_image,
                first_ocr_result,
                info,
            )

        info["selectedCorrectionDegrees"] = round(best_angle, 3)
        info["selectedScore"] = best_score
        for candidate_info in info["candidates"]:
            candidate_info["selected"] = bool(
                float(candidate_info.get("correctionDegrees", 0.0))
                == best_angle
            )
        cv2.imwrite(str(selected_debug_path), best_card)
        return (
            best_card,
            best_enhanced,
            best_result,
            info,
        )

    def select_better_full_ocr_result(
        self,
        card_result: dict[str, Any],
        enhanced_result: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        """Chỉ chọn enhanced khi không làm mất bằng chứng của ảnh gốc."""
        card_score = self.score_full_ocr_result(card_result)
        enhanced_score = self.score_full_ocr_result(enhanced_result)

        card_evidence = self.summarize_full_ocr_evidence(card_result)
        enhanced_evidence = self.summarize_full_ocr_evidence(
            enhanced_result
        )

        # Confidence/nhãn tăng nhẹ không được phép đổi lấy việc mất trường
        # lõi hoặc mất nhiều dòng. Đây là nguyên nhân bản adaptive trước có
        # thể chọn một ảnh trông "điểm cao" nhưng Layout Parser đọc kém hơn.
        if (
            enhanced_evidence["validCoreFields"]
            < card_evidence["validCoreFields"]
        ):
            return card_result, "card"
        if (
            card_evidence["meaningfulLines"] >= 3
            and enhanced_evidence["meaningfulLines"]
            < card_evidence["meaningfulLines"]
        ):
            return card_result, "card"

        if enhanced_score > card_score + 0.75:
            return enhanced_result, "enhanced"
        return card_result, "card"

    def summarize_full_ocr_evidence(
        self,
        ocr_result: dict[str, Any],
    ) -> dict[str, int]:
        """Đếm trường hợp lệ và dòng có nghĩa để chống chọn nhầm ảnh."""
        structured = ocr_result.get("structuredData", {})
        if not isinstance(structured, dict):
            structured = {}
        field_checks = (
            self.validator.is_valid_id_number(
                structured.get("idNumber")
            ),
            self.validator.is_valid_name(
                structured.get("fullName")
            ),
            self.validator.is_valid_date(
                structured.get("dateOfBirth")
            ),
            self.validator.is_valid_gender(
                structured.get("gender")
            ),
            self.validator.is_valid_nationality(
                structured.get("nationality")
            ),
        )
        lines = ocr_result.get(
            "normalizedText",
            ocr_result.get("rawText", []),
        )
        if isinstance(lines, str):
            lines = lines.splitlines()
        if not isinstance(lines, list):
            lines = []
        meaningful_lines = sum(
            bool(re.search(r"[A-Za-zÀ-ỹ0-9]{3}", str(line)))
            for line in lines
        )
        return {
            "validCoreFields": sum(bool(value) for value in field_checks),
            "meaningfulLines": meaningful_lines,
        }

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
        address_layout: dict[str, Any] | None = None,
        field_layout: dict[str, Any] | None = None,
        reference_data: dict[str, Any] | None = None,
        skip_fields: Collection[str] | None = None,
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
                    address_layout=address_layout,
                    field_layout=field_layout,
                    reference_data=reference_data,
                    skip_fields=skip_fields,
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

    def build_parser_diagnostics(
        self,
        merged_data: dict[str, Any],
        data_sources: dict[str, str],
        field_confidences: dict[str, float],
        validation_result: dict[str, Any],
        card_side: dict[str, Any],
        qr_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        field_validity = validation_result.get("fieldValidity", {})
        if not isinstance(field_validity, dict):
            field_validity = {}

        fields: dict[str, dict[str, Any]] = {}
        valid_fields: list[str] = []
        missing_fields: list[str] = []
        invalid_fields: list[str] = []
        for field_name in self.FIELD_NAMES:
            value = merged_data.get(field_name)
            present = bool(str(value or "").strip())
            valid = bool(field_validity.get(field_name, False))
            if valid:
                valid_fields.append(field_name)
            elif present:
                invalid_fields.append(field_name)
            else:
                missing_fields.append(field_name)
            fields[field_name] = {
                "present": present,
                "valid": valid,
                "source": str(
                    data_sources.get(field_name, "NOT_FOUND")
                ),
                "confidence": round(
                    float(field_confidences.get(field_name, 0.0) or 0.0),
                    4,
                ),
            }

        if validation_result.get("isValid"):
            status = "COMPLETE"
        elif valid_fields:
            status = "PARTIAL"
        else:
            status = "FAILED"

        return {
            "status": status,
            "fields": fields,
            "validFields": valid_fields,
            "missingFields": missing_fields,
            "invalidFields": invalid_fields,
            "errors": list(validation_result.get("errors", []) or []),
            "cardSide": card_side,
            "qrStatus": qr_metadata.get("status"),
            "qrRegionDetected": bool(
                qr_metadata.get("regionDetected")
            ),
            "qrDecoded": bool(qr_metadata.get("decoded")),
        }

    @staticmethod
    def attach_qr_region_to_field_layout(
        field_layout: dict[str, Any] | None,
        qr_result: dict[str, Any],
        card_size: tuple[int, int],
    ) -> dict[str, Any]:
        """Đưa QR vào cùng sơ đồ 1000x630 với các field parser."""
        layout = dict(field_layout or {})
        regions = dict(layout.get("regions") or {})
        card_width = max(int(card_size[0]), 1)
        card_height = max(int(card_size[1]), 1)
        scale_x = 1000.0 / card_width
        scale_y = 630.0 / card_height

        box = qr_result.get("boundingBox")
        detected = bool(qr_result.get("regionDetected"))
        source = "QR_DETECTOR"
        if not isinstance(box, dict):
            box = qr_result.get("searchRegion")
            detected = False
            source = "QR_SEARCH_REGION"
        if isinstance(box, dict):
            x1 = int(round(float(box.get("x", 0)) * scale_x))
            y1 = int(round(float(box.get("y", 0)) * scale_y))
            x2 = int(round(
                (float(box.get("x", 0)) + float(box.get("width", 0)))
                * scale_x
            ))
            y2 = int(round(
                (float(box.get("y", 0)) + float(box.get("height", 0)))
                * scale_y
            ))
            normalized_box = {
                "x1": max(0, min(1000, x1)),
                "y1": max(0, min(630, y1)),
                "x2": max(0, min(1000, x2)),
                "y2": max(0, min(630, y2)),
            }
            regions["qrCode"] = {
                "field": normalized_box,
                "value": normalized_box,
                "detected": detected,
                "decoded": bool(qr_result.get("decoded")),
                "source": source,
            }
        layout["regions"] = regions
        return layout

    def public_qr_metadata(
        self,
        qr_result: dict[str, Any],
        qr_fusion: dict[str, Any],
        skipped_fields: Collection[str],
    ) -> dict[str, Any]:
        """Trả metadata QR cần thiết mà không lặp lại payload chứa PII."""
        status = str(qr_result.get("status") or "").strip()
        if not status:
            if qr_result.get("decoded"):
                status = "DECODED_VALID"
            elif (
                qr_result.get("regionDetected")
                or qr_result.get("qrRegionDetected")
            ):
                status = "DETECTED_NOT_DECODED"
            else:
                status = "NOT_DETECTED"
        conflicts = [
            {
                "field": str(item.get("field")),
                "ocrSource": str(item.get("ocrSource", "NOT_FOUND")),
                "resolution": str(item.get("resolution", "CCCD_QR")),
                "requiresReview": bool(item.get("requiresReview", True)),
            }
            for item in qr_fusion.get("conflicts", [])
            if isinstance(item, dict) and item.get("field")
        ]
        auxiliary = qr_result.get("auxiliaryData", {})
        if not isinstance(auxiliary, dict):
            auxiliary = {}
        return self.make_json_safe(
            {
                "enabled": bool(settings.qr_fast_path_enabled),
                "status": status,
                "message": qr_result.get("statusMessage"),
                "decoded": bool(qr_result.get("decoded")),
                "payloadDecoded": bool(qr_result.get("payloadDecoded")),
                "used": bool(qr_fusion.get("used")),
                "decoder": qr_result.get("decoder"),
                "selectedDecoder": qr_result.get("selectedDecoder"),
                "attemptCount": int(
                    qr_result.get("attemptCount", 0) or 0
                ),
                "elapsedMs": float(qr_result.get("elapsedMs", 0.0) or 0.0),
                "selectedVariant": qr_result.get("selectedVariant"),
                "detectionVariant": qr_result.get("detectionVariant"),
                "regionDetected": bool(
                    qr_result.get("regionDetected")
                    or qr_result.get("qrRegionDetected")
                ),
                "polygon": qr_result.get("polygon", []),
                "boundingBox": qr_result.get("boundingBox"),
                "searchRegion": qr_result.get("searchRegion"),
                "searchRegions": qr_result.get("searchRegions", {}),
                "orientationRotationDegrees": int(
                    qr_result.get("orientationRotationDegrees", 0) or 0
                ),
                "orientationProbeAttemptCount": int(
                    qr_result.get("orientationProbeAttemptCount", 0) or 0
                ),
                "orientationProbeElapsedMs": float(
                    qr_result.get("orientationProbeElapsedMs", 0.0) or 0.0
                ),
                "format": qr_result.get("format"),
                "fieldCount": int(qr_result.get("fieldCount", 0) or 0),
                "providedFields": list(
                    qr_result.get("providedFields", []) or []
                ),
                "missingRequiredFields": list(
                    qr_result.get("missingRequiredFields", []) or []
                ),
                "appliedFields": list(
                    qr_fusion.get("appliedFields", []) or []
                ),
                "agreementFields": list(
                    qr_fusion.get("agreementFields", []) or []
                ),
                "conflicts": conflicts,
                "skippedFieldOcr": sorted(
                    str(field_name) for field_name in skipped_fields
                ),
                "hasOldDocumentNumber": bool(
                    auxiliary.get("hasOldDocumentNumber")
                ),
                "hasDateOfIssue": bool(auxiliary.get("hasDateOfIssue")),
                "additionalFieldCount": int(
                    auxiliary.get("additionalFieldCount", 0) or 0
                ),
                "errors": list(qr_result.get("errors", []) or []),
                "errorDetails": list(
                    qr_result.get("errorDetails", []) or []
                ),
                "debug": dict(qr_result.get("debug", {}) or {}),
            }
        )

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

        for metadata_key in (
            "dataSources",
            "qrFastPath",
            "parserDiagnostics",
            "imageQuality",
            "cardSide",
            "geometry",
            "orientation",
            "processingStagesMs",
            "debugDir",
            "cardImage",
            "enhancedImage",
            "reviewRequired",
            "fieldDebug",
            "resizeRatio",
        ):
            if metadata_key in partial:
                response["metadata"][metadata_key] = self.make_json_safe(
                    partial.get(metadata_key)
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
