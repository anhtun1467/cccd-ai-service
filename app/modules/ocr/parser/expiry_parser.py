from __future__ import annotations

import re
import unicodedata
from datetime import datetime


class ExpiryParser:
    """
    Parser trích xuất ngày hết hạn của CCCD.
    Đã cải tiến hỗ trợ loại bỏ dấu tiếng Việt để nhận diện nhãn chính xác hơn.
    """

    DATE_PATTERN = re.compile(
        r"\b("
        r"\d{1,2}"
        r"[/-]"
        r"\d{1,2}"
        r"[/-]"
        r"\d{4}"
        r")\b"
    )

    EXPIRY_LABEL_PATTERN = re.compile(
        r"(?:"
        r"co\s*gia\s*tri\s*den"
        r"|gia\s*tri\s*den"
        r"|date\s*of\s*expiry"
        r"|expiry\s*date"
        r"|date\s*of\s*expir[yv]"
        r")",
        re.IGNORECASE,
    )

    def _remove_diacritics(self, text: str) -> str:
        """Loại bỏ dấu tiếng Việt để tăng độ chính xác khi so khớp nhãn OCR."""
        if not text:
            return ""
        nfkd = unicodedata.normalize("NFKD", text)
        return "".join([c for c in nfkd if not unicodedata.combining(c)])

    def parse(
        self,
        lines: list[str],
    ) -> str | None:
        """
        Trích xuất ngày hết hạn từ danh sách dòng OCR.
        """
        for index, line in enumerate(lines):
            normalized_line = self._remove_diacritics(line)
            if not self.EXPIRY_LABEL_PATTERN.search(normalized_line):
                continue

            date_value = self.extract_date(line)

            if date_value:
                return date_value

            # Trường hợp ngày hết hạn nằm ở dòng kế tiếp.
            if index + 1 < len(lines):
                date_value = self.extract_date(
                    lines[index + 1]
                )

                if date_value:
                    return date_value

        return None

    def extract_date(
        self,
        text: str,
    ) -> str | None:
        """
        Tìm và chuẩn hóa ngày tháng theo định dạng DD/MM/YYYY.
        """
        match = self.DATE_PATTERN.search(text)

        if not match:
            return None

        date_text = match.group(1).replace("-", "/")

        try:
            parsed_date = datetime.strptime(
                date_text,
                "%d/%m/%Y",
            )

            return parsed_date.strftime("%d/%m/%Y")

        except ValueError:
            return None