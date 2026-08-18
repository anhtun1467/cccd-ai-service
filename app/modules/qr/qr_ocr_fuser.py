from __future__ import annotations

import re
import unicodedata
from copy import deepcopy
from typing import Any

from app.modules.ocr.validator import CCCDValidator


QR_MAPPABLE_FIELDS: tuple[str, ...] = (
    "idNumber",
    "fullName",
    "dateOfBirth",
    "gender",
    "placeOfResidence",
)


def fuse_qr_data(
    ocr_data: dict[str, Any],
    ocr_sources: dict[str, str],
    qr_data: dict[str, Any] | None,
    validator: CCCDValidator,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    result = deepcopy(ocr_data)
    sources = deepcopy(ocr_sources)
    diagnostics: dict[str, Any] = {
        "used": False,
        "appliedFields": [],
        "agreementFields": [],
        "conflicts": [],
        "ignoredFields": [],
    }
    qr_values = qr_data or {}

    for field_name in QR_MAPPABLE_FIELDS:
        qr_value = qr_values.get(field_name)
        if not is_valid_field_value(field_name, qr_value, validator):
            if qr_value:
                diagnostics["ignoredFields"].append(field_name)
            continue

        current_value = result.get(field_name)
        current_source = sources.get(field_name, "NOT_FOUND")
        if current_value:
            if values_equivalent(field_name, current_value, qr_value):
                diagnostics["agreementFields"].append(field_name)
            elif is_valid_field_value(
                field_name,
                current_value,
                validator,
            ):
                diagnostics["conflicts"].append(
                    {
                        "field": field_name,
                        "ocrSource": current_source,
                        "resolution": "CCCD_QR",
                        # Địa chỉ QR là chuỗi dữ liệu máy đọc được và thường
                        # đầy đủ hơn OCR nhiều dòng. Khác biệt địa chỉ vẫn
                        # được giữ để chẩn đoán, nhưng không biến một kết quả
                        # đủ 8 trường thành OCR_PARTIAL/lỗi chất lượng ảnh.
                        "requiresReview": field_name != "placeOfResidence",
                    }
                )

        result[field_name] = qr_value
        sources[field_name] = "CCCD_QR"
        diagnostics["appliedFields"].append(field_name)

    diagnostics["used"] = bool(diagnostics["appliedFields"])
    return result, sources, diagnostics


def select_qr_field_ocr_skips(
    qr_data: dict[str, Any] | None,
    full_card_data: dict[str, Any] | None,
    validator: CCCDValidator,
) -> set[str]:
    """Bỏ OCR field đã được QR xác nhận; QR địa chỉ luôn là nguồn chính."""
    skips: set[str] = set()
    qr_values = qr_data or {}
    full_values = full_card_data or {}
    for field_name in QR_MAPPABLE_FIELDS:
        qr_value = qr_values.get(field_name)
        if not is_valid_field_value(field_name, qr_value, validator):
            continue
        if field_name == "placeOfResidence":
            skips.add(field_name)
            continue
        full_value = full_values.get(field_name)
        if not full_value or values_equivalent(
            field_name,
            full_value,
            qr_value,
        ):
            skips.add(field_name)
    return skips


def build_qr_reference_data(
    full_card_data: dict[str, Any] | None,
    qr_data: dict[str, Any] | None,
    validator: CCCDValidator,
) -> dict[str, Any]:
    reference = dict(full_card_data or {})
    for field_name in QR_MAPPABLE_FIELDS:
        value = (qr_data or {}).get(field_name)
        if is_valid_field_value(field_name, value, validator):
            reference[field_name] = value
    return reference


def is_valid_field_value(
    field_name: str,
    value: Any,
    validator: CCCDValidator,
) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if field_name == "idNumber":
        return validator.is_valid_id_number(text)
    if field_name == "fullName":
        return validator.is_valid_name(text)
    if field_name == "dateOfBirth":
        return validator.is_valid_date(text)
    if field_name == "gender":
        return validator.is_valid_gender(text)
    if field_name == "placeOfResidence":
        return validator.is_valid_address(
            text,
            field_name="placeOfResidence",
        )
    return False


def values_equivalent(
    field_name: str,
    first: Any,
    second: Any,
) -> bool:
    if field_name in {"idNumber", "dateOfBirth", "gender"}:
        return str(first or "").strip().casefold() == str(
            second or ""
        ).strip().casefold()
    return normalized_text_key(first) == normalized_text_key(second)


def normalized_text_key(value: Any) -> str:
    text = unicodedata.normalize("NFD", str(value or ""))
    plain = "".join(
        character
        for character in text
        if unicodedata.category(character) != "Mn"
    )
    plain = plain.replace("đ", "d").replace("Đ", "D")
    return re.sub(r"[^a-z0-9]", "", plain.casefold())
