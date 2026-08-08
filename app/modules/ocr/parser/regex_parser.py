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
        r"|\bof\s*birth"
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
        """Trích xuất họ tên ở trước hoặc sau nhãn họ tên."""

        for index, line in enumerate(lines):
            match = self.NAME_LABEL_PATTERN.search(line)
            if not match:
                continue

            # Dữ liệu thường nằm sau nhãn hoặc ở dòng kế tiếp. Phần trước
            # nhãn chỉ là phương án cuối vì OCR có thể đọc sai nhãn Việt
            # thành "Ho vaa ten / Full name" và biến nó thành họ tên giả.
            candidates = [line[match.end():]]
            if index + 1 < len(lines):
                candidates.append(lines[index + 1])
            if index + 2 < len(lines):
                candidates.append(lines[index + 2])

            prefix = line[:match.start()]
            plain_prefix = OCRTextNormalizer._remove_accents(
                prefix
            ).lower()
            if not re.search(
                r"\bho\s*va{1,2}\s*te?n\b",
                plain_prefix,
            ):
                candidates.append(prefix)

            if index > 0:
                candidates.append(lines[index - 1])

            for candidate in candidates:
                value = self.clean_full_name(candidate)
                if value:
                    return value

        return None

    def clean_full_name(
        self,
        value: str | None,
    ) -> str | None:
        if not value:
            return None

        value = self.remove_known_labels(value)
        value = re.sub(r"[^A-Za-zÀ-ỹ\s]", " ", value)
        words = [
            word.upper()
            for word in re.sub(r"\s+", " ", value).strip().split()
            if len(word) > 1
        ]

        forbidden = {
            "HO", "VA", "TEN", "FULL", "NAME",
            "NGAY", "SINH", "DATE", "BIRTH",
            "GIOI", "TINH", "SEX", "QUOC",
            "TICH", "NATIONALITY", "CITIZEN",
            "IDENTITY", "CARD",
        }
        words = [word for word in words if word not in forbidden]

        if not 2 <= len(words) <= 7:
            return None

        return " ".join(words)

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
        """Trích xuất giới tính mà không nhầm chữ Nam trong Việt Nam."""

        for index, line in enumerate(lines):
            match = self.GENDER_LABEL_PATTERN.search(line)
            if not match:
                continue

            section = line[match.end():]
            section = self.NATIONALITY_LABEL_PATTERN.split(
                section,
                maxsplit=1,
            )[0]
            section = re.sub(
                r"^\s*/?\s*sex\s*[:;/,-]?\s*",
                "",
                section,
                flags=re.IGNORECASE,
            )

            candidates = [section]
            if index + 1 < len(lines):
                candidates.append(lines[index + 1])

            for candidate in candidates:
                plain = OCRTextNormalizer.normalize(candidate).lower()
                tokens = re.findall(r"[a-zà-ỹ]+", plain)

                if any(token in {"nu", "nữ", "ni", "nv", "nw", "female"} for token in tokens):
                    return "Nữ"
                if any(token in {"nam", "male"} for token in tokens):
                    return "Nam"

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
                return "Việt Nam"

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
                return "Việt Nam"

        return None

    def parse_address_field(
        self,
        lines: list[str],
        label_pattern: re.Pattern[str],
    ) -> str | None:
        """Trích xuất địa chỉ, loại nhãn và vùng ngày hết hạn bị ghép nhầm."""

        for index, line in enumerate(lines):
            match = label_pattern.search(line)
            if not match:
                continue

            values: list[str] = []
            before = line[:match.start()].strip(" :;,/-\"")
            after = line[match.end():].strip(" :;,/-\"")

            # Chỉ dùng phần trước nhãn khi thực sự giống địa chỉ.
            if self.looks_like_address(before):
                values.append(before)
            if self.looks_like_address(after):
                values.append(after)

            next_index = index + 1
            while next_index < len(lines) and len(values) < 3:
                next_line = lines[next_index].strip()
                next_index += 1

                if not next_line:
                    continue
                if self.STOP_LABEL_PATTERN.search(next_line):
                    break
                if self.ID_PATTERN.search(next_line.replace(" ", "")):
                    break
                if self.looks_like_address(next_line):
                    values.append(next_line)

            address = self.clean_address(", ".join(values))
            return address or None

        return None

    @staticmethod
    def looks_like_address(value: str | None) -> bool:
        if not value:
            return False

        text = re.sub(
            r"(?:co|c[o0])\s*[g9]ia\s*(?:tri|t[1il])\s*den\s*"
            r"\d{1,2}[/-]\d{1,2}[/-]\d{4}",
            " ",
            value,
            flags=re.IGNORECASE,
        )
        tokens = re.findall(r"[A-Za-zÀ-ỹ0-9]+", text)
        return len(tokens) >= 2 and not all(len(token) == 1 for token in tokens)

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
            r"(?:co|c[o0])\s*[g9]ia\s*(?:tri|t[1il])\s*den\s*"
            r"\d{1,2}[/-]\d{1,2}[/-]\d{4}",
            " ",
            value,
            flags=re.IGNORECASE,
        )

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
