import re
from datetime import datetime
from typing import Any


class CCCDValidator:
    """
    Kiểm tra dữ liệu CCCD sau OCR Parser.
    """

    def validate(self, data: dict[str, Any]) -> dict[str, Any]:
        errors = []

        if not self.is_valid_id_number(data.get("idNumber")):
            errors.append("Số CCCD không hợp lệ")

        if not self.is_valid_name(data.get("fullName")):
            errors.append("Họ tên không hợp lệ")

        if not self.is_valid_date(data.get("dateOfBirth")):
            errors.append("Ngày sinh không hợp lệ")

        if not self.is_valid_gender(data.get("gender")):
            errors.append("Giới tính không hợp lệ")

        if not self.is_valid_nationality(data.get("nationality")):
            errors.append("Quốc tịch không hợp lệ")

        return {
            "isValid": len(errors) == 0,
            "errors": errors,
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
