import re
from typing import Any


class CCCDRegexParser:
    """
    Parser dữ liệu CCCD từ raw OCR text.
    """

    def parse(self, raw_text: list[str]) -> dict[str, Any]:
        cleaned_lines = [self.clean_line(line) for line in raw_text if line.strip()]
        full_text = " ".join(cleaned_lines)

        return {
            "idNumber": self.extract_id_number(full_text),
            "fullName": self.extract_full_name(cleaned_lines),
            "dateOfBirth": self.extract_date_of_birth(full_text),
            "gender": self.extract_gender(full_text),
            "nationality": self.extract_nationality(full_text),
            "rawText": cleaned_lines,
        }

    def clean_line(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        text = text.replace("!", "/")
        text = text.replace("|", "/")
        text = text.replace(";", ":")
        return text

    def extract_id_number(self, text: str) -> str | None:
        match = re.search(r"\b\d{12}\b", text)
        return match.group(0) if match else None

    def extract_date_of_birth(self, text: str) -> str | None:
        text = self.fix_date_text(text)

        match = re.search(r"\b\d{2}/\d{2}/\d{4}\b", text)
        return match.group(0) if match else None

    def extract_gender(self, text: str) -> str | None:
        upper_text = text.upper()

        if re.search(r"\bNAM\b", upper_text):
            return "Nam"

        if "NU" in upper_text or "NỮ" in upper_text:
            return "Nữ"

        return None

    def extract_nationality(self, text: str) -> str | None:
        upper_text = text.upper()

        if "VIET" in upper_text or "VICT" in upper_text or "NANA" in upper_text:
            return "Việt Nam"

        return None

    def extract_full_name(self, lines: list[str]) -> str | None:
        for index, line in enumerate(lines):
            upper_line = line.upper()

            if (
                "FULL NAME" in upper_line
                or "HOVA TEN" in upper_line
                or "HO VA TEN" in upper_line
            ):
                if index + 1 < len(lines):
                    candidate = lines[index + 1].strip()
                    return self.normalize_name(candidate)

        for line in lines:
            candidate = line.strip()
            upper_candidate = candidate.upper()

            if self.looks_like_name(upper_candidate):
                return self.normalize_name(candidate)

        return None

    def looks_like_name(self, text: str) -> bool:
        if not text:
            return False

        if any(char.isdigit() for char in text):
            return False

        words = text.split()

        if len(words) < 2:
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

        for keyword in ignored_keywords:
            if keyword in text:
                return False

        return text.isupper()

    def normalize_name(self, name: str) -> str:
        name = re.sub(r"[^A-Za-zÀ-ỹ\s]", "", name)
        name = re.sub(r"\s+", " ", name)
        return name.strip().upper()

    def fix_date_text(self, text: str) -> str:
        text = text.replace("O", "0")
        text = text.replace("o", "0")
        text = text.replace("I", "1")
        text = text.replace("l", "1")

        # 24/0311995 -> 24/03/1995
        text = re.sub(
            r"(\d{2})/(\d{2})1(\d{4})",
            r"\1/\2/\3",
            text,
        )

        # 24/031995 -> 24/03/1995
        text = re.sub(
            r"(\d{2})/(\d{2})(\d{4})",
            r"\1/\2/\3",
            text,
        )

        return text
