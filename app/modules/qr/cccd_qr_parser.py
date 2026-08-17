from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from typing import Any


class CCCDQRParser:
    """Parse payload QR CCCD mà không suy đoán trường không có trong QR."""

    MINIMUM_FIELD_COUNT = 7
    MAXIMUM_FIELD_COUNT = 11
    REQUIRED_FIELDS = (
        "idNumber",
        "fullName",
        "dateOfBirth",
        "gender",
    )

    def parse(self, payload: str | None) -> dict[str, Any]:
        cleaned_payload = self.clean_payload(payload)
        empty_result = {
            "success": False,
            "format": None,
            "fieldCount": 0,
            "structuredData": {},
            "auxiliaryData": {},
            "providedFields": [],
            "errors": [],
        }
        if not cleaned_payload:
            return {
                **empty_result,
                "errors": ["QR_PAYLOAD_EMPTY"],
            }

        fields = cleaned_payload.split("|")
        field_count = len(fields)
        if not (
            self.MINIMUM_FIELD_COUNT
            <= field_count
            <= self.MAXIMUM_FIELD_COUNT
        ):
            return {
                **empty_result,
                "fieldCount": field_count,
                "errors": ["QR_FIELD_COUNT_UNSUPPORTED"],
            }

        identifier = self.clean_text(fields[0])
        old_document_number = self.clean_text(fields[1])
        full_name = self.clean_text(fields[2])
        date_of_birth = self.normalize_date(fields[3])
        gender = self.normalize_gender(fields[4])
        residence = self.clean_address(fields[5])
        date_of_issue = self.normalize_date(fields[6])

        structured_data: dict[str, str] = {}
        errors: list[str] = []

        if re.fullmatch(r"\d{12}", identifier):
            structured_data["idNumber"] = identifier
        else:
            errors.append("QR_ID_NUMBER_INVALID")

        if self.is_valid_name(full_name):
            structured_data["fullName"] = full_name
        else:
            errors.append("QR_FULL_NAME_INVALID")

        if date_of_birth and self.is_birth_date(date_of_birth):
            structured_data["dateOfBirth"] = date_of_birth
        else:
            errors.append("QR_DATE_OF_BIRTH_INVALID")

        if gender:
            structured_data["gender"] = gender
        else:
            errors.append("QR_GENDER_INVALID")

        if self.is_plausible_residence(residence):
            structured_data["placeOfResidence"] = residence

        auxiliary_data: dict[str, Any] = {
            "hasOldDocumentNumber": bool(
                re.fullmatch(r"\d{9}|\d{12}", old_document_number)
            ),
            "hasDateOfIssue": bool(date_of_issue),
            "additionalFieldCount": max(0, field_count - 7),
        }
        if old_document_number and not auxiliary_data[
            "hasOldDocumentNumber"
        ]:
            errors.append("QR_OLD_DOCUMENT_NUMBER_INVALID")
        if fields[6].strip() and not date_of_issue:
            errors.append("QR_DATE_OF_ISSUE_INVALID")

        missing_required = [
            field_name
            for field_name in self.REQUIRED_FIELDS
            if not structured_data.get(field_name)
        ]
        success = not missing_required
        format_name = (
            "CCCD_QR_7_FIELDS"
            if field_count == 7
            else "CAN_CUOC_QR_EXTENDED"
        )
        return {
            "success": success,
            "format": format_name,
            "fieldCount": field_count,
            "structuredData": structured_data if success else {},
            "auxiliaryData": auxiliary_data,
            "providedFields": (
                list(structured_data.keys()) if success else []
            ),
            "errors": errors,
        }

    @staticmethod
    def clean_payload(payload: str | None) -> str:
        if payload is None:
            return ""
        text = unicodedata.normalize("NFC", str(payload))
        text = text.replace("\ufeff", "").replace("\x00", "")
        return text.strip(" \t\r\n")

    @staticmethod
    def clean_text(value: str | None) -> str:
        text = unicodedata.normalize("NFC", str(value or ""))
        text = "".join(
            character
            for character in text
            if unicodedata.category(character) not in {"Cc", "Cf"}
        )
        return re.sub(r"\s+", " ", text).strip(" ,;:/-")

    @classmethod
    def clean_address(cls, value: str | None) -> str:
        text = cls.clean_text(value)
        text = re.sub(r"\s*([,;])\s*", r"\1 ", text)
        return re.sub(r"\s+", " ", text).strip(" ,;")

    @staticmethod
    def normalize_date(value: str | None) -> str | None:
        text = re.sub(r"\s+", "", str(value or ""))
        if not text:
            return None
        formats = (
            "%d%m%Y",
            "%d/%m/%Y",
            "%d-%m-%Y",
            "%d.%m.%Y",
            "%Y-%m-%d",
        )
        for date_format in formats:
            try:
                parsed = datetime.strptime(text, date_format)
            except ValueError:
                continue
            if 1900 <= parsed.year <= 2100:
                return parsed.strftime("%d/%m/%Y")
        return None

    @staticmethod
    def normalize_gender(value: str | None) -> str | None:
        text = unicodedata.normalize("NFD", str(value or ""))
        plain = "".join(
            character
            for character in text
            if unicodedata.category(character) != "Mn"
        ).casefold()
        plain = re.sub(r"[^a-z]", "", plain)
        if plain in {"nam", "male", "m"}:
            return "Nam"
        if plain in {"nu", "female", "f"}:
            return "Nữ"
        return None

    @staticmethod
    def is_valid_name(value: str) -> bool:
        words = [word for word in value.split() if word]
        return bool(
            2 <= len(words) <= 8
            and all(any(character.isalpha() for character in word) for word in words)
        )

    @staticmethod
    def is_birth_date(value: str) -> bool:
        try:
            parsed = datetime.strptime(value, "%d/%m/%Y").date()
        except ValueError:
            return False
        return date(1900, 1, 1) <= parsed <= date.today()

    @staticmethod
    def is_plausible_residence(value: str) -> bool:
        words = re.findall(r"[^\W_]+", value, flags=re.UNICODE)
        return bool(len(value) >= 8 and len(words) >= 3)
