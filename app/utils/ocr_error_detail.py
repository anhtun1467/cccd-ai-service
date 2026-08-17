from __future__ import annotations

from typing import Any


DEFAULT_SUGGESTIONS: dict[str, str] = {
    "MULTIPLE_CARDS": "Chỉ đặt một CCCD trong ảnh rồi chụp lại.",
    "CCCD_BACK_SIDE_DETECTED": (
        "Chụp mặt trước CCCD có số định danh, họ tên và ảnh chân dung."
    ),
    "CARD_DETECTION_FAILED": (
        "Đặt trọn bốn góc CCCD trong khung, tránh tay hoặc vật khác che cạnh."
    ),
    "OCR_CORE_FIELDS_MISSING": (
        "Kiểm tra ảnh cắt thẻ và các khung parser trong thư mục debug."
    ),
    "OCR_CORE_FIELDS_MISSING_LOW_QUALITY": (
        "Giữ máy ổn định, lấy nét vào chữ và tăng ánh sáng rồi chụp lại."
    ),
}


def build_ocr_error_detail(result: dict[str, Any]) -> dict[str, Any]:
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    rejection = metadata.get("rejection")
    if not isinstance(rejection, dict):
        rejection = {}
    validation = metadata.get("validation")
    if not isinstance(validation, dict):
        validation = {}
    parser_diagnostics = metadata.get("parserDiagnostics")
    if not isinstance(parser_diagnostics, dict):
        parser_diagnostics = {}

    error_code = str(rejection.get("errorCode") or "OCR_FAILED")
    message = str(
        result.get("message")
        or rejection.get("reason")
        or "OCR CCCD thất bại"
    )
    reason = str(rejection.get("reason") or message)
    suggestion = str(
        rejection.get("suggestion")
        or DEFAULT_SUGGESTIONS.get(
            error_code,
            "Xem chi tiết lỗi và ảnh debug trước khi chụp lại.",
        )
    )

    missing_fields = rejection.get("missingCoreFields")
    if not isinstance(missing_fields, list):
        missing_fields = parser_diagnostics.get("missingFields", [])
    detected_fields = rejection.get("readableCoreFields")
    if not isinstance(detected_fields, list):
        detected_fields = parser_diagnostics.get("validFields", [])

    return {
        "message": message,
        "error_code": error_code,
        "stage": rejection.get("stage", "OCR_PIPELINE"),
        "reason": reason,
        "suggestion": suggestion,
        "image_quality": metadata.get("imageQuality"),
        "card_side": metadata.get("cardSide"),
        "detected_fields": detected_fields,
        "missing_fields": missing_fields,
        "validation_errors": list(validation.get("errors", []) or []),
        "qr": metadata.get("qrFastPath"),
        "parser": parser_diagnostics or None,
        "card_count": rejection.get("cardCount"),
        "debug_dir": metadata.get("debugDir"),
        "debug_images": rejection.get("debugImages"),
    }
