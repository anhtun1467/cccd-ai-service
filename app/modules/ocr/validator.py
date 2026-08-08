import re
import unicodedata
from datetime import datetime
from typing import Any

from app.modules.ocr.result_fuser import (
    is_valid_address as is_plausible_address,
)


class CCCDValidator:
    """
    Kiểm tra dữ liệu CCCD sau OCR Parser.
    """

    def validate(self, data: dict[str, Any]) -> dict[str, Any]:
        field_validity = {
            "idNumber": self.is_valid_id_number(data.get("idNumber")),
            "fullName": self.is_valid_name(data.get("fullName")),
            "dateOfBirth": self.is_valid_date(data.get("dateOfBirth")),
            "gender": self.is_valid_gender(data.get("gender")),
            "nationality": self.is_valid_nationality(
                data.get("nationality")
            ),
            "placeOfOrigin": self.is_valid_address(
                data.get("placeOfOrigin"),
                field_name="placeOfOrigin",
            ),
            "placeOfResidence": self.is_valid_address(
                data.get("placeOfResidence"),
                field_name="placeOfResidence",
            ),
            "dateOfExpiry": self.is_valid_date(data.get("dateOfExpiry")),
        }
        error_messages = {
            "idNumber": "Số CCCD không hợp lệ",
            "fullName": "Họ tên không hợp lệ",
            "dateOfBirth": "Ngày sinh không hợp lệ",
            "gender": "Giới tính không hợp lệ",
            "nationality": "Quốc tịch không hợp lệ",
            "placeOfOrigin": "Quê quán thiếu hoặc không đáng tin cậy",
            "placeOfResidence": (
                "Nơi thường trú thiếu hoặc không đáng tin cậy"
            ),
            "dateOfExpiry": "Ngày hết hạn không hợp lệ",
        }
        errors = [
            error_messages[field_name]
            for field_name, is_valid in field_validity.items()
            if not is_valid
        ]

        return {
            "isValid": len(errors) == 0,
            "errors": errors,
            "fieldValidity": field_validity,
        }

    def is_valid_id_number(self, value: str | None) -> bool:
        return bool(value and re.fullmatch(r"\d{12}", value))

    def is_valid_name(self, value: str | None) -> bool:
        return bool(value and len(value.split()) >= 2)

    def is_valid_date(self, value: str | None) -> bool:
        if not value:
            return False

        try:
            datetime.strptime(value, "%d/%m/%Y")
            return True
        except ValueError:
            return False

    def is_valid_gender(self, value: str | None) -> bool:
        return value in {"Nam", "Nữ"}

    def is_valid_nationality(self, value: str | None) -> bool:
        return value in {"Viet Nam", "Việt Nam"}

    def is_valid_address(
        self,
        value: str | None,
        field_name: str = "placeOfResidence",
    ) -> bool:
        """Chỉ xác nhận địa chỉ đủ cấu trúc, không đoán địa danh bị mờ."""
        if not value:
            return False
        if not is_plausible_address(value, field_name):
            return False

        text = re.sub(r"\s+", " ", str(value)).strip(" ,;:/-")
        words = re.findall(r"[^\W_]+", text, flags=re.UNICODE)
        components = [
            item.strip()
            for item in re.split(r"[,;]", text)
            if item.strip()
        ]
        if len(words) < 3 or len(components) < 2:
            return False

        plain = self._plain_text(text)
        forbidden_labels = (
            "place of origin",
            "place of residence",
            "date of expiry",
            "co gia tri den",
            "gia tri den",
            "ngay sinh",
            "date of birth",
            "nationality",
        )
        if any(label in plain for label in forbidden_labels):
            return False

        # Chuỗi số dài hoặc ngày bị dính giữa địa chỉ là dấu hiệu hai cột
        # OCR đã bị ghép nhầm. Không kết luận hợp lệ trong trường hợp này.
        if re.search(r"\d{5,}|\d{1,8}/\d{4}", plain):
            return False

        single_character_words = sum(len(word) == 1 for word in words)
        if (
            single_character_words >= 3
            and single_character_words / len(words) >= 0.25
        ):
            return False

        final_words = re.findall(
            r"[^\W_]+",
            components[-1],
            flags=re.UNICODE,
        )
        if any(len(word) == 1 for word in final_words):
            return False

        return True

    @staticmethod
    def _plain_text(value: str) -> str:
        normalized = unicodedata.normalize("NFD", value)
        text = "".join(
            character
            for character in normalized
            if unicodedata.category(character) != "Mn"
        )
        return text.replace("đ", "d").replace("Đ", "D").lower()
