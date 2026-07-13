import re


class NameParser:
    """
    Tách họ tên từ OCR lines.
    """

    def parse(self, lines: list[str]) -> str | None:
        for index, line in enumerate(lines):
            upper_line = line.upper()

            if (
                "FULL NAME" in upper_line
                or "HO VA TEN" in upper_line
                or "HOVA TEN" in upper_line
            ):
                if index + 1 < len(lines):
                    return self.normalize_name(lines[index + 1])

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
        ]

        return text.isupper() and not any(
            keyword in text for keyword in ignored_keywords
        )

    def normalize_name(self, name: str) -> str:
        name = re.sub(r"[^A-Za-zÀ-ỹ\s]", "", name)
        name = re.sub(r"\s+", " ", name)
        return name.strip().upper()