import re


class BirthParser:
    """
    Tách ngày sinh.
    """

    def parse(self, text: str) -> str | None:
        match = re.search(
            r"\b\d{2}/\d{2}/\d{4}\b",
            text,
        )

        if match:
            return match.group(0)

        return None