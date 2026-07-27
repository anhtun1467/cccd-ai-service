from __future__ import annotations

import re
from datetime import datetime


class ExpiryParser:
    """
    Parser trích xuất ngày hết hạn của CCCD.
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

    def parse(
        self,
        lines: list[str],
    ) -> str | None:
        """
        Trích xuất ngày hết hạn từ danh sách dòng OCR.
        """

        for index, line in enumerate(lines):
            if not self.EXPIRY_LABEL_PATTERN.search(line):
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
