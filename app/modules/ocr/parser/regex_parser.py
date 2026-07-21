from __future__ import annotations

import re
from typing import Any

from app.modules.ocr.parser.expiry_parser import ExpiryParser
from app.modules.ocr.text_normalizer import OCRTextNormalizer


class CCCDRegexParser:
    """
    Parser tổng hợp thông tin CCCD từ các dòng OCR đã chuẩn hóa.

    Các trường được trích xuất:
    - Số CCCD
    - Họ và tên
    - Ngày sinh
    - Giới tính
    - Quốc tịch
    - Quê quán
    - Nơi thường trú
    - Ngày hết hạn
    """

    ID_PATTERN = re.compile(
        r"(?<!\d)(\d{12})(?!\d)"
    )

    DATE_PATTERN = re.compile(
        r"\b("
        r"\d{1,2}"
        r"[/-]"
        r"\d{1,2}"
        r"[/-]"
        r"\d{4}"
        r")\b"
    )

    NAME_LABEL_PATTERN = re.compile(
        r"(?:"
        r"ho\s*va\s*ten"
        r"|full\s*name"
        r")",
        re.IGNORECASE,
    )

    BIRTH_LABEL_PATTERN = re.compile(
        r"(?:"
        r"ngay\s*sinh"
        r"|date\s*of\s*birth"
        r")",
        re.IGNORECASE,
    )

    GENDER_LABEL_PATTERN = re.compile(
        r"(?:"
        r"gioi\s*tinh"
        r"|sex"
        r")",
        re.IGNORECASE,
    )

    NATIONALITY_LABEL_PATTERN = re.compile(
        r"(?:"
        r"quoc\s*tich"
        r"|nationality"
        r")",
        re.IGNORECASE,
    )

    ORIGIN_LABEL_PATTERN = re.compile(
        r"(?:"
        r"que\s*quan"
        r"|place\s*of\s*origin"
        r")",
        re.IGNORECASE,
    )

    RESIDENCE_LABEL_PATTERN = re.compile(
        r"(?:"
        r"noi\s*thuong\s*tru"
        r"|place\s*of\s*residence"
        r")",
        re.IGNORECASE,
    )

    STOP_LABEL_PATTERN = re.compile(
        r"(?:"
        r"ho\s*va\s*ten"
        r"|full\s*name"
        r"|ngay\s*sinh"
        r"|date\s*of\s*birth"
        r"|gioi\s*tinh"
        r"|sex"
        r"|quoc\s*tich"
        r"|nationality"
        r"|que\s*quan"
        r"|place\s*of\s*origin"
        r"|noi\s*thuong\s*tru"
        r"|place\s*of\s*residence"
        r"|date\s*of\s*expiry"
        r"|co\s*gia\s*tri\s*den"
        r")",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        self.expiry_parser = ExpiryParser()

    def parse(
        self,
        text: list[str] | str,
    ) -> dict[str, Any]:
        """
        Parse dữ liệu CCCD từ chuỗi hoặc danh sách dòng OCR.
        """

        lines = self.prepare_lines(text)

        return {
            "idNumber": self.parse_id_number(lines),
            "fullName": self.parse_full_name(lines),
            "dateOfBirth": self.parse_date_of_birth(lines),
            "gender": self.parse_gender(lines),
            "nationality": self.parse_nationality(lines),
            "placeOfOrigin": self.parse_address_field(
                lines=lines,
                label_pattern=self.ORIGIN_LABEL_PATTERN,
            ),
            "placeOfResidence": self.parse_address_field(
                lines=lines,
                label_pattern=self.RESIDENCE_LABEL_PATTERN,
            ),
            "dateOfExpiry": self.expiry_parser.parse(lines),
        }

    @staticmethod
    def prepare_lines(
        text: list[str] | str,
    ) -> list[str]:
        """
        Chuẩn hóa dữ liệu đầu vào thành danh sách dòng.
        """

        if isinstance(text, str):
            source_lines = text.splitlines()
        else:
            source_lines = text

        return OCRTextNormalizer.normalize_lines(
            [
                str(line)
                for line in source_lines
                if line is not None
            ]
        )

    def parse_id_number(
        self,
        lines: list[str],
    ) -> str | None:
        """
        Tìm số CCCD gồm 12 chữ số.
        """

        for line in lines:
            compact_line = re.sub(
                r"(?<=\d)\s+(?=\d)",
                "",
                line,
            )

            match = self.ID_PATTERN.search(
                compact_line
            )

            if match:
                return match.group(1)

        return None

    def parse_full_name(
        self,
        lines: list[str],
    ) -> str | None:
        """
        Trích xuất họ tên.
        """

        value = self.extract_label_value(
            lines=lines,
            label_pattern=self.NAME_LABEL_PATTERN,
            allow_next_line=True,
        )

        if not value:
            return None

        value = self.remove_known_labels(value)
        value = re.sub(
            r"[^A-Za-zÀ-ỹ\s]",
            " ",
            value,
        )
        value = re.sub(r"\s+", " ", value).strip()

        if not value:
            return None

        return value.upper()

    def parse_date_of_birth(
        self,
        lines: list[str],
    ) -> str | None:
        """
        Trích xuất ngày sinh.
        """

        for index, line in enumerate(lines):
            if not self.BIRTH_LABEL_PATTERN.search(line):
                continue

            date_value = self.extract_date(line)

            if date_value:
                return date_value

            if index + 1 < len(lines):
                date_value = self.extract_date(
                    lines[index + 1]
                )

                if date_value:
                    return date_value

        return None

    def parse_gender(
        self,
        lines: list[str],
    ) -> str | None:
        """
        Trích xuất giới tính.
        """

        value = self.extract_label_value(
            lines=lines,
            label_pattern=self.GENDER_LABEL_PATTERN,
            allow_next_line=True,
        )

        if not value:
            return None

        normalized = value.lower()

        if re.search(r"\b(nam|male)\b", normalized):
            return "Nam"

        if re.search(r"\b(nu|nữ|female)\b", normalized):
            return "Nữ"

        return None

    def parse_nationality(
        self,
        lines: list[str],
    ) -> str | None:
        """
        Trích xuất quốc tịch.
        """

        value = self.extract_label_value(
            lines=lines,
            label_pattern=self.NATIONALITY_LABEL_PATTERN,
            allow_next_line=True,
        )

        if value:
            normalized = value.lower()

            if re.search(
                r"\b(viet\s*nam|vict\s*nam|vict\s*nana)\b",
                normalized,
            ):
                return "Viet Nam"

            value = self.remove_known_labels(value)
            value = re.sub(
                r"[^A-Za-zÀ-ỹ\s]",
                " ",
                value,
            )
            value = re.sub(r"\s+", " ", value).strip()

            if value:
                return value

        # CCCD Việt Nam gần như luôn có quốc tịch Việt Nam.
        for line in lines:
            if re.search(
                r"\bviet\s*nam\b",
                line,
                re.IGNORECASE,
            ):
                return "Viet Nam"

        return None

    def parse_address_field(
        self,
        lines: list[str],
        label_pattern: re.Pattern[str],
    ) -> str | None:
        """
        Trích xuất quê quán hoặc nơi thường trú.

        Có thể ghép thêm một dòng tiếp theo nếu chưa gặp nhãn mới.
        """

        for index, line in enumerate(lines):
            if not label_pattern.search(line):
                continue

            first_value = self.remove_label(
                line,
                label_pattern,
            )

            values: list[str] = []

            if first_value:
                values.append(first_value)

            next_index = index + 1

            while next_index < len(lines):
                next_line = lines[next_index].strip()

                if not next_line:
                    next_index += 1
                    continue

                if self.STOP_LABEL_PATTERN.search(next_line):
                    break

                if self.ID_PATTERN.search(
                    next_line.replace(" ", "")
                ):
                    break

                values.append(next_line)

                # Địa chỉ trên CCCD thường chỉ kéo dài tối đa hai dòng.
                if len(values) >= 2:
                    break

                next_index += 1

            address = ", ".join(values)
            address = self.clean_address(address)

            return address or None

        return None

    def extract_label_value(
        self,
        lines: list[str],
        label_pattern: re.Pattern[str],
        allow_next_line: bool = True,
    ) -> str | None:
        """
        Lấy phần dữ liệu nằm sau nhãn.

        Ví dụ:
        Ho va ten / Full name: NGUYEN VAN A
        """

        for index, line in enumerate(lines):
            if not label_pattern.search(line):
                continue

            value = self.remove_label(
                line,
                label_pattern,
            )

            if value:
                return value

            if allow_next_line and index + 1 < len(lines):
                next_line = lines[index + 1].strip()

                if (
                    next_line
                    and not self.STOP_LABEL_PATTERN.search(next_line)
                ):
                    return next_line

        return None

    @staticmethod
    def remove_label(
        line: str,
        label_pattern: re.Pattern[str],
    ) -> str:
        """
        Xóa nhãn và các ký tự phân cách khỏi dòng OCR.
        """

        value = label_pattern.sub(
            "",
            line,
            count=1,
        )

        # Xóa nhãn tiếng Anh còn sót lại sau dấu "/".
        value = re.sub(
            r"^\s*/?\s*"
            r"(?:"
            r"full\s*name"
            r"|date\s*of\s*birth"
            r"|sex"
            r"|nationality"
            r"|place\s*of\s*origin"
            r"|place\s*of\s*residence"
            r")"
            r"\s*:?\s*",
            "",
            value,
            flags=re.IGNORECASE,
        )

        value = re.sub(
            r"^[\s:;/,-]+",
            "",
            value,
        )

        return value.strip()

    def remove_known_labels(
        self,
        value: str,
    ) -> str:
        """
        Xóa các nhãn CCCD còn sót trong giá trị.
        """

        value = self.STOP_LABEL_PATTERN.sub(
            " ",
            value,
        )

        value = re.sub(
            r"^[\s:;/,-]+",
            "",
            value,
        )

        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def clean_address(
        value: str,
    ) -> str:
        """
        Làm sạch chuỗi địa chỉ.
        """

        value = re.sub(
            r"\s+([,.;])",
            r"\1",
            value,
        )

        value = re.sub(
            r"[,;]\s*[,;]+",
            ", ",
            value,
        )

        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        return value.strip(" ,.;:-")

    def extract_date(
        self,
        text: str,
    ) -> str | None:
        """
        Trích xuất ngày tháng và chuẩn hóa thành DD/MM/YYYY.
        """

        match = self.DATE_PATTERN.search(text)

        if not match:
            return None

        date_value = match.group(1).replace("-", "/")

        parts = date_value.split("/")

        if len(parts) != 3:
            return None

        day, month, year = parts

        try:
            day_number = int(day)
            month_number = int(month)
            year_number = int(year)

            if not 1 <= day_number <= 31:
                return None

            if not 1 <= month_number <= 12:
                return None

            if year_number < 1900:
                return None

        except ValueError:
            return None

        return (
            f"{day_number:02d}/"
            f"{month_number:02d}/"
            f"{year_number:04d}"
        )