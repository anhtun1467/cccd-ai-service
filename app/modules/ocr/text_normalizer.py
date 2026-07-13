from __future__ import annotations

import re


class TextNormalizer:
    """
    Chuẩn hóa kết quả OCR trước khi parser.
    """

    COMMON_REPLACEMENTS = {
        "Vict Nana": "Viet Nam",
        "VICT NANA": "VIET NAM",
        "Netonelt": "Nationality",
        "Gioitinh": "Gioi tinh",
        "Hova ten": "Ho va ten",
        "Ngay sinh": "Ngay sinh",
        "Ha No": "Ha Noi",
        "Can Cuoc Conc Dan": "Can Cuoc Cong Dan",
        "CONC DAN": "CONG DAN",
    }

    def normalize(self, lines: list[str]) -> list[str]:
        normalized = []

        for line in lines:
            line = self.normalize_spaces(line)
            line = self.fix_common_words(line)
            line = self.fix_date(line)

            normalized.append(line)

        return normalized

    def normalize_spaces(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def fix_common_words(self, text: str) -> str:
        for wrong, correct in self.COMMON_REPLACEMENTS.items():
            text = text.replace(wrong, correct)

        return text

    def fix_date(self, text: str) -> str:
        text = text.replace("!", "/")
        text = text.replace("|", "/")
        text = text.replace(";", ":")

        text = re.sub(
            r"(\d{2})/(\d{2})1(\d{4})",
            r"\1/\2/\3",
            text,
        )

        text = re.sub(
            r"(\d{2})/(\d{2})(\d{4})",
            r"\1/\2/\3",
            text,
        )

        return text