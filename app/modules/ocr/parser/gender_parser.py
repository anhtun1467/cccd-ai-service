from __future__ import annotations

import re
import unicodedata


class GenderParser:
    """
    Parser tách giới tính từ văn bản OCR.
    Hỗ trợ chuẩn hóa bỏ dấu để nhận diện nhãn linh hoạt hơn.
    """

    def _remove_diacritics(self, text: str) -> str:
        if not text:
            return ""
        nfkd = unicodedata.normalize("NFKD", text)
        return "".join([c for c in nfkd if not unicodedata.combining(c)])

    def parse(self, text: str) -> str | None:
        if not text:
            return None

        normalized = self._remove_diacritics(text).upper()

        if re.search(r"\bNAM\b", normalized):
            return "Nam"

        if re.search(r"\bNU\b", normalized):
            return "Nữ"

        return None