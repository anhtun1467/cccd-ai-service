from __future__ import annotations

from app.utils.ocr_error_detail import build_ocr_error_detail


def test_sharp_parser_failure_does_not_ask_user_to_improve_focus() -> None:
    detail = build_ocr_error_detail({
        "status": "OCR_FAILED",
        "message": "Parser chưa xác nhận được đủ trường",
        "metadata": {
            "rejection": {
                "errorCode": "OCR_CORE_FIELDS_MISSING",
                "stage": "PARSER_VALIDATION",
                "reason": "Ảnh đủ nét nhưng vùng cắt chưa đúng",
                "readableCoreFields": ["idNumber", "fullName"],
                "missingCoreFields": ["dateOfBirth"],
            },
            "imageQuality": {
                "cropBlurScore": 519.72,
                "decision": "PASSED_IMAGE_FAILED_PARSER",
            },
            "validation": {
                "isValid": False,
                "errors": ["Ngày sinh không hợp lệ"],
            },
            "debugDir": "storage/debug/example",
        },
    })

    assert detail["error_code"] == "OCR_CORE_FIELDS_MISSING"
    assert detail["stage"] == "PARSER_VALIDATION"
    assert detail["image_quality"]["decision"] == (
        "PASSED_IMAGE_FAILED_PARSER"
    )
    assert "lấy nét" not in detail["suggestion"].casefold()
    assert detail["missing_fields"] == ["dateOfBirth"]

