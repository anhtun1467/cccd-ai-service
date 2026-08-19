from __future__ import annotations

from typing import Any


QR_ERROR_CATALOG: dict[str, dict[str, Any]] = {
    "QR_IMAGE_INVALID": {
        "stage": "input",
        "message": "Ảnh dùng để tìm QR rỗng hoặc không hợp lệ.",
        "retryable": False,
    },
    "QR_REGION_NOT_DETECTED": {
        "stage": "detection",
        "message": (
            "Không khoanh được vùng QR. OCR chữ vẫn tiếp tục bình thường."
        ),
        "retryable": True,
    },
    "QR_REGION_DETECTED_NOT_DECODED": {
        "stage": "decode",
        "message": (
            "Đã khoanh đúng vùng QR nhưng chưa giải mã được dữ liệu. "
            "OCR chữ vẫn được dùng làm phương án dự phòng."
        ),
        "retryable": True,
    },
    "QR_PAYLOAD_EMPTY": {
        "stage": "parse",
        "message": "QR giải mã thành chuỗi rỗng.",
        "retryable": True,
    },
    "QR_FIELD_COUNT_UNSUPPORTED": {
        "stage": "parse",
        "message": "QR không có cấu trúc 7 đến 11 trường của CCCD.",
        "retryable": False,
    },
    "QR_ID_NUMBER_INVALID": {
        "stage": "parse",
        "message": "Số định danh trong QR không đủ 12 chữ số.",
        "retryable": False,
    },
    "QR_FULL_NAME_INVALID": {
        "stage": "parse",
        "message": "Họ tên trong QR không hợp lệ.",
        "retryable": False,
    },
    "QR_DATE_OF_BIRTH_INVALID": {
        "stage": "parse",
        "message": "Ngày sinh trong QR không hợp lệ.",
        "retryable": False,
    },
    "QR_GENDER_INVALID": {
        "stage": "parse",
        "message": "Giới tính trong QR không hợp lệ.",
        "retryable": False,
    },
    "QR_OLD_DOCUMENT_NUMBER_INVALID": {
        "stage": "parse",
        "message": "Số giấy tờ cũ trong QR không đúng định dạng.",
        "retryable": False,
    },
    "QR_DATE_OF_ISSUE_INVALID": {
        "stage": "parse",
        "message": "Ngày cấp trong QR không đúng định dạng.",
        "retryable": False,
    },
    "NON_CCCD_QR_IGNORED": {
        "stage": "parse",
        "message": "Đã đọc được QR nhưng nội dung không phải QR CCCD hợp lệ.",
        "retryable": False,
    },
    "QR_TIME_BUDGET_REACHED": {
        "stage": "decode",
        "message": "Đã dừng thử QR để không làm chậm toàn bộ OCR.",
        "retryable": True,
    },
    "QR_FAST_PATH_DISABLED": {
        "stage": "configuration",
        "message": "Nhánh đọc QR đang bị tắt trong cấu hình.",
        "retryable": False,
    },
    "QR_DECODER_UNAVAILABLE": {
        "stage": "configuration",
        "message": "Không có bộ giải mã QR khả dụng.",
        "retryable": False,
    },
    "QR_DECODE_ERROR": {
        "stage": "decode",
        "message": "Bộ giải mã QR gặp lỗi ở một biến thể ảnh.",
        "retryable": True,
    },
    "QR_FAST_PATH_ERROR": {
        "stage": "internal",
        "message": "Nhánh QR gặp lỗi nội bộ; OCR chữ vẫn tiếp tục.",
        "retryable": True,
    },
}


def normalize_qr_error_code(value: Any) -> str:
    code = str(value or "").strip()
    if not code:
        return "QR_UNKNOWN_ERROR"
    return code.split(":", 1)[0]


def build_qr_error_details(errors: list[Any] | tuple[Any, ...]) -> list[dict]:
    details: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_error in errors:
        code = normalize_qr_error_code(raw_error)
        if code in seen:
            continue
        seen.add(code)
        catalog_item = QR_ERROR_CATALOG.get(code)
        if catalog_item is None:
            catalog_item = {
                "stage": "unknown",
                "message": "Không xác định được nguyên nhân lỗi QR.",
                "retryable": True,
            }
        details.append({"code": code, **catalog_item})
    return details


def derive_qr_status(result: dict[str, Any]) -> str:
    if result.get("decoded"):
        return "DECODED_VALID"
    if result.get("payloadDecoded"):
        return "DECODED_NON_CCCD"
    if result.get("regionDetected") or result.get("qrRegionDetected"):
        return "DETECTED_NOT_DECODED"
    if "QR_FAST_PATH_DISABLED" in result.get("errors", []):
        return "DISABLED"
    if any(
        normalize_qr_error_code(error) == "QR_FAST_PATH_ERROR"
        for error in result.get("errors", [])
    ):
        return "ERROR"
    return "NOT_DETECTED"


def qr_status_message(status: str) -> str:
    return {
        "DECODED_VALID": "Đã khoanh và giải mã QR CCCD thành công.",
        "DECODED_NON_CCCD": (
            "Đã khoanh và đọc QR nhưng payload không đúng cấu trúc CCCD."
        ),
        "DETECTED_NOT_DECODED": (
            "Đã khoanh vùng QR nhưng chưa giải mã được; OCR chữ vẫn chạy."
        ),
        "DISABLED": "Nhánh QR đang tắt; hệ thống chỉ sử dụng OCR chữ.",
        "ERROR": "Nhánh QR gặp lỗi; hệ thống đã chuyển sang OCR chữ.",
        "NOT_DETECTED": (
            "Chưa khoanh được QR; hệ thống tiếp tục bằng OCR chữ."
        ),
    }.get(status, "Không xác định được trạng thái QR.")

