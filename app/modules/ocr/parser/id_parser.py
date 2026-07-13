import re


class IdParser:
    """
    Tách số CCCD từ OCR text.
    """

    def parse(self, text: str) -> str | None:
        match = re.search(r"\b\d{12}\b", text)
        return match.group(0) if match else None