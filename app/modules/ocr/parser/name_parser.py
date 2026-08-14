from __future__ import annotations

import re


class NameParser:
    """
    Parser tách họ tên từ các dòng OCR.
    """

    def parse(self, lines: list[str]) -> str | None:
        if not lines:
            return None

        for index, line in enumerate(lines):
            upper_line = line.upper()

            if any(
                keyword in upper_line
                for keyword in [
                    "FULL NAME",
                    "HO VA TEN",
                    "HOVA TEN",
                ]
            ):
                if index + 1 < len(lines):
                    normalized = self.normalize_name(lines[index + 1])
                    if normalized:
                        return normalized

        for line in lines:
            candidate = line.strip()
            upper_candidate = candidate.upper()

            if self.looks_like_name(upper_candidate):
                return self.normalize_name(candidate)

        return None

    def looks_like_name(self, text: str) -> bool:
        if not text or any(char.isdigit() for char in text):
            return False

        if len(text.split()) < 2:
            return False

        ignored_keywords = [
            "CONG HOA",
            "SOCIALIST",
            "INDEPENDENCE",
            "FREEDOM",
            "HAPPINESS",
            "CAN CUOC",
            "CITIZEN",
            "FULL NAME",
            "DATE OF BIRTH",
            "PLACE OF",
            "RESIDENCE",
            "EXPIRY",
            "VIET NAM",
            "QUOC GIA",
        ]

        return text.isupper() and not any(
            keyword in text for keyword in ignored_keywords
        )

    def normalize_name(self, name: str) -> str | None:
        if not name:
            return None
        name = re.sub(r"[^A-Za-zÀ-ỹ\s]", "", name)
        name = re.sub(r"\s+", " ", name)
        cleaned = name.strip().upper()
        
        # Đảm bảo tên hợp lệ có ít nhất 2 từ
        if len(cleaned.split()) >= 2:
            return cleaned
        return None