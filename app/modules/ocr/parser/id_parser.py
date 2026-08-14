from __future__ import annotations

import re


class IdParser:
    """
    Parser tách số CCCD (12 chữ số) từ OCR text.
    Hỗ trợ xử lý loại bỏ khoảng trắng hoặc ký tự phân cách vô ý.
    """

    def parse(self, text: str) -> str | None:
        if not text:
            return None

        # Loại bỏ khoảng trắng hoặc dấu gạch ngang giữa các cụm số nếu OCR quét nhầm
        cleaned_text = re.sub(r"[\s\-]", "", text)

        match = re.search(r"\b\d{12}\b", cleaned_text)
        return match.group(0) if match else None