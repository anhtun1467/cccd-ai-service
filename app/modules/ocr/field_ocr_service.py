from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from app.modules.ocr.base_engine import BaseOCREngine
from app.modules.ocr.easyocr_engine import EasyOCREngine
from app.modules.ocr.field_cropper import CCCDFieldCropper
from app.modules.ocr.text_normalizer import OCRTextNormalizer


class FieldOCRService:
    """
    OCR từng vùng thông tin trên mặt trước CCCD.

    Quy trình:
        Card image
        -> Chuẩn hóa ảnh CCCD
        -> Cắt từng trường
        -> Phát hiện và cắt ảnh chân dung
        -> OCR từng trường chữ
        -> Chuẩn hóa văn bản
        -> Làm sạch theo từng loại dữ liệu
        -> Trả về structured data
    """

    OCR_FIELDS: tuple[str, ...] = (
        "idNumber",
        "fullName",
        "dateOfBirth",
        "gender",
        "nationality",
        "placeOfOrigin",
        "placeOfResidence",
        "dateOfExpiry",
    )

    DATE_PATTERN = re.compile(
        r"(?<!\d)"
        r"(\d{1,2})"
        r"\s*[/.\-]\s*"
        r"(\d{1,2})"
        r"\s*[/.\-]\s*"
        r"(\d{4})"
        r"(?!\d)"
    )

    FIELD_LABEL_PATTERNS: dict[str, tuple[str, ...]] = {
        "idNumber": (
            r"\bso\b",
            r"\bno\b",
            r"\bnumber\b",
        ),
        "fullName": (
            r"\bho\s*va\s*ten\b",
            r"\bfull\s*name\b",
            r"\bhova\s*ten\b",
        ),
        "dateOfBirth": (
            r"\bngay\s*sinh\b",
            r"\bdate\s*of\s*birth\b",
        ),
        "gender": (
            r"\bgioi\s*tinh\b",
            r"\bsex\b",
        ),
        "nationality": (
            r"\bquoc\s*tich\b",
            r"\bnationality\b",
        ),
        "placeOfOrigin": (
            r"\bque\s*quan\b",
            r"\bplace\s*of\s*origin\b",
        ),
        "placeOfResidence": (
            r"\bnoi\s*thuong\s*tru\b",
            r"\bplace\s*of\s*residence\b",
        ),
        "dateOfExpiry": (
            r"\bco\s*gia\s*tri\s*den\b",
            r"\bdate\s*of\s*expiry\b",
            r"\bexpiry\s*date\b",
        ),
    }

    def __init__(
        self,
        engine: BaseOCREngine | None = None,
    ) -> None:
        self.engine = engine or EasyOCREngine()
        self.cropper = CCCDFieldCropper()

    def extract_fields(
        self,
        card_image_path: str,
        output_dir: str,
    ) -> dict[str, Any]:
        """
        Cắt ảnh CCCD và OCR từng trường riêng biệt.

        Args:
            card_image_path:
                Đường dẫn ảnh CCCD đã được detect và perspective transform.

            output_dir:
                Thư mục lưu ảnh field, ảnh portrait và ảnh debug.

        Returns:
            Dictionary gồm:
            - structuredData
            - fieldResults
            - portrait
            - debug
        """

        card_path = Path(card_image_path)

        if not card_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy ảnh CCCD: {card_path}"
            )

        field_results = self.cropper.crop_fields_from_path(
            image_path=str(card_path),
            output_dir=output_dir,
        )

        structured_data: dict[str, str | None] = {
            field_name: None
            for field_name in self.OCR_FIELDS
        }

        field_ocr_results: dict[str, dict[str, Any]] = {}

        for field_name in self.OCR_FIELDS:
            field_result = field_results.get(field_name)

            if not field_result:
                field_ocr_results[field_name] = (
                    self.build_empty_field_result(
                        field_name=field_name,
                        message="Không tìm thấy vùng cắt",
                    )
                )
                continue

            processed_image_path = field_result.get(
                "imagePath"
            )

            raw_image_path = field_result.get(
                "rawImagePath"
            )

            if not processed_image_path:
                field_ocr_results[field_name] = (
                    self.build_empty_field_result(
                        field_name=field_name,
                        message="Không có đường dẫn ảnh field",
                        raw_image_path=raw_image_path,
                    )
                )
                continue

            processed_path = Path(
                str(processed_image_path)
            )

            if not processed_path.exists():
                field_ocr_results[field_name] = (
                    self.build_empty_field_result(
                        field_name=field_name,
                        message=(
                            "Không tìm thấy ảnh field: "
                            f"{processed_path}"
                        ),
                        image_path=str(processed_path),
                        raw_image_path=raw_image_path,
                    )
                )
                continue

            try:
                ocr_result = self.engine.recognize(
                    str(processed_path)
                )
            except Exception as error:
                field_ocr_results[field_name] = (
                    self.build_empty_field_result(
                        field_name=field_name,
                        message=f"OCR thất bại: {error}",
                        image_path=str(processed_path),
                        raw_image_path=raw_image_path,
                    )
                )
                continue

            normalized_lines = (
                OCRTextNormalizer.normalize_lines(
                    ocr_result.raw_text
                )
            )

            joined_text = self.join_field_text(
                field_name=field_name,
                lines=normalized_lines,
            )

            cleaned_value = self.clean_field_value(
                field_name=field_name,
                value=joined_text,
            )

            average_confidence = (
                self.calculate_average_confidence(
                    ocr_result.text_boxes
                )
            )

            structured_data[field_name] = cleaned_value

            field_ocr_results[field_name] = {
                "fieldName": field_name,
                "success": bool(ocr_result.success),
                "message": ocr_result.message,
                "rawText": list(ocr_result.raw_text),
                "normalizedText": normalized_lines,
                "joinedText": joined_text,
                "value": cleaned_value,
                "averageConfidence": average_confidence,
                "imagePath": str(processed_path),
                "rawImagePath": (
                    str(raw_image_path)
                    if raw_image_path
                    else None
                ),
                "box": field_result.get("box"),
                "rawWidth": field_result.get("rawWidth"),
                "rawHeight": field_result.get("rawHeight"),
                "processedWidth": field_result.get(
                    "processedWidth"
                ),
                "processedHeight": field_result.get(
                    "processedHeight"
                ),
            }

        portrait_result = self.prepare_portrait_result(
            field_results.get("portrait")
        )

        return {
            "structuredData": structured_data,
            "fieldResults": field_ocr_results,
            "portrait": portrait_result,
            "debug": field_results.get("_debug"),
        }

    def prepare_portrait_result(
        self,
        portrait_result: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """
        Chuẩn hóa dữ liệu ảnh chân dung.

        Ảnh portrait không được OCR. Ảnh này được dùng cho module
        Face Verification ở bước tiếp theo.
        """

        if not portrait_result:
            return None

        image_path = portrait_result.get("imagePath")
        raw_image_path = portrait_result.get(
            "rawImagePath"
        )

        image_exists = bool(
            image_path
            and Path(str(image_path)).exists()
        )

        raw_image_exists = bool(
            raw_image_path
            and Path(str(raw_image_path)).exists()
        )

        return {
            "success": image_exists,
            "faceDetected": bool(
                portrait_result.get("faceDetected", False)
            ),
            "imagePath": (
                str(image_path)
                if image_path
                else None
            ),
            "rawImagePath": (
                str(raw_image_path)
                if raw_image_path
                else None
            ),
            "imageExists": image_exists,
            "rawImageExists": raw_image_exists,
            "box": portrait_result.get("box"),
            "rawWidth": portrait_result.get("rawWidth"),
            "rawHeight": portrait_result.get("rawHeight"),
            "processedWidth": portrait_result.get(
                "processedWidth"
            ),
            "processedHeight": portrait_result.get(
                "processedHeight"
            ),
        }

    @staticmethod
    def join_field_text(
        field_name: str,
        lines: list[str],
    ) -> str:
        """
        Ghép các dòng OCR của một field.

        Địa chỉ được ghép bằng dấu phẩy.
        Các trường còn lại được ghép bằng khoảng trắng.
        """

        clean_lines = [
            line.strip()
            for line in lines
            if line and line.strip()
        ]

        if not clean_lines:
            return ""

        if field_name in {
            "placeOfOrigin",
            "placeOfResidence",
        }:
            return ", ".join(clean_lines)

        return " ".join(clean_lines)

    def clean_field_value(
        self,
        field_name: str,
        value: str,
    ) -> str | None:
        """
        Làm sạch giá trị OCR theo loại trường.
        """

        if not value:
            return None

        cleaned = OCRTextNormalizer.normalize(value)

        cleaned = self.remove_field_labels(
            field_name=field_name,
            value=cleaned,
        )

        if not cleaned:
            return None

        if field_name == "idNumber":
            return self.clean_id_number(cleaned)

        if field_name == "fullName":
            return self.clean_full_name(cleaned)

        if field_name in {
            "dateOfBirth",
            "dateOfExpiry",
        }:
            return self.clean_date(cleaned)

        if field_name == "gender":
            return self.clean_gender(cleaned)

        if field_name == "nationality":
            return self.clean_nationality(cleaned)

        if field_name in {
            "placeOfOrigin",
            "placeOfResidence",
        }:
            return self.clean_address(cleaned)

        return cleaned or None

    def remove_field_labels(
        self,
        field_name: str,
        value: str,
    ) -> str:
        """
        Xóa các nhãn tiếng Việt và tiếng Anh còn sót lại trong vùng OCR.
        """

        patterns = self.FIELD_LABEL_PATTERNS.get(
            field_name,
            (),
        )

        cleaned = value

        for pattern in patterns:
            cleaned = re.sub(
                pattern,
                " ",
                cleaned,
                flags=re.IGNORECASE,
            )

        cleaned = re.sub(
            r"^[\s:/,;.\-]+",
            "",
            cleaned,
        )

        cleaned = re.sub(
            r"\s+",
            " ",
            cleaned,
        ).strip()

        return cleaned

    @staticmethod
    def clean_id_number(
        value: str,
    ) -> str | None:
        """
        Làm sạch số CCCD.

        Chỉ giữ chữ số và yêu cầu đúng 12 số.
        """

        substitutions = str.maketrans(
            {
                "O": "0",
                "o": "0",
                "I": "1",
                "i": "1",
                "L": "1",
                "l": "1",
            }
        )

        normalized = value.translate(substitutions)

        digits = re.sub(
            r"\D",
            "",
            normalized,
        )

        match = re.search(
            r"(?<!\d)(\d{12})(?!\d)",
            digits,
        )

        if match:
            return match.group(1)

        return None

    @staticmethod
    def clean_full_name(
        value: str,
    ) -> str | None:
        """
        Làm sạch họ tên.

        Giữ chữ cái tiếng Việt và khoảng trắng.
        """

        cleaned = re.sub(
            r"[^A-Za-zÀ-ỹ\s]",
            " ",
            value,
        )

        cleaned = re.sub(
            r"\s+",
            " ",
            cleaned,
        ).strip()

        if len(cleaned) < 3:
            return None

        words = cleaned.split()

        if len(words) < 2:
            return None

        return cleaned.upper()

    def clean_date(
        self,
        value: str,
    ) -> str | None:
        """
        Chuẩn hóa ngày về DD/MM/YYYY và kiểm tra ngày hợp lệ.
        """

        normalized = value.translate(
            str.maketrans(
                {
                    "O": "0",
                    "o": "0",
                    "I": "1",
                    "i": "1",
                    "l": "1",
                    "h": "/",
                }
            )
        )

        match = self.DATE_PATTERN.search(
            normalized
        )

        if not match:
            digits = re.sub(
                r"\D",
                "",
                normalized,
            )

            if len(digits) == 8:
                normalized = (
                    f"{digits[0:2]}/"
                    f"{digits[2:4]}/"
                    f"{digits[4:8]}"
                )

                match = self.DATE_PATTERN.search(
                    normalized
                )

        if not match:
            return None

        day = int(match.group(1))
        month = int(match.group(2))
        year = int(match.group(3))

        try:
            parsed_date = datetime(
                year=year,
                month=month,
                day=day,
            )
        except ValueError:
            return None

        return parsed_date.strftime("%d/%m/%Y")

    @staticmethod
    def clean_gender(
        value: str,
    ) -> str | None:
        """
        Chuẩn hóa giới tính về Nam hoặc Nữ.
        """

        lowered = value.lower()

        if re.search(
            r"\b(nam|male)\b",
            lowered,
        ):
            return "Nam"

        if re.search(
            r"\b(nữ|nu|female)\b",
            lowered,
        ):
            return "Nữ"

        return None

    @staticmethod
    def clean_nationality(
        value: str,
    ) -> str | None:
        """
        Chuẩn hóa quốc tịch.
        """

        lowered = value.lower()

        vietnam_patterns = (
            r"\bviet\s*nam\b",
            r"\bvietnam\b",
            r"\bvict\s*nam\b",
            r"\bviet\s*nana\b",
            r"\bvict\s*nana\b",
        )

        for pattern in vietnam_patterns:
            if re.search(pattern, lowered):
                return "Viet Nam"

        cleaned = re.sub(
            r"[^A-Za-zÀ-ỹ\s]",
            " ",
            value,
        )

        cleaned = re.sub(
            r"\s+",
            " ",
            cleaned,
        ).strip()

        return cleaned or None

    @staticmethod
    def clean_address(
        value: str,
    ) -> str | None:
        """
        Làm sạch quê quán hoặc nơi thường trú.
        """

        cleaned = value.replace(";", ",")

        cleaned = re.sub(
            r"\s+([,.:])",
            r"\1",
            cleaned,
        )

        cleaned = re.sub(
            r"[,]\s*[,]+",
            ", ",
            cleaned,
        )

        cleaned = re.sub(
            r"\s*,\s*",
            ", ",
            cleaned,
        )

        cleaned = re.sub(
            r"\s+",
            " ",
            cleaned,
        )

        cleaned = cleaned.strip(
            " ,.;:/-"
        )

        if len(cleaned) < 3:
            return None

        return cleaned

    @staticmethod
    def calculate_average_confidence(
        text_boxes: list[Any],
    ) -> float:
        """
        Tính confidence trung bình của kết quả OCR.
        """

        if not text_boxes:
            return 0.0

        scores: list[float] = []

        for item in text_boxes:
            confidence = getattr(
                item,
                "confidence",
                None,
            )

            if confidence is None and isinstance(
                item,
                dict,
            ):
                confidence = item.get(
                    "confidence",
                    0.0,
                )

            try:
                score = float(confidence)
            except (TypeError, ValueError):
                continue

            if 0.0 <= score <= 1.0:
                scores.append(score)

        if not scores:
            return 0.0

        return round(
            sum(scores) / len(scores),
            4,
        )

    @staticmethod
    def build_empty_field_result(
        field_name: str,
        message: str,
        image_path: str | None = None,
        raw_image_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Tạo kết quả mặc định khi một field không OCR được.
        """

        return {
            "fieldName": field_name,
            "success": False,
            "message": message,
            "rawText": [],
            "normalizedText": [],
            "joinedText": "",
            "value": None,
            "averageConfidence": 0.0,
            "imagePath": image_path,
            "rawImagePath": raw_image_path,
            "box": None,
        }


field_ocr_service = FieldOCRService()