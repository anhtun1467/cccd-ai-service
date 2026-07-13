import re


class GenderParser:
    """
    Tách giới tính.
    """

    def parse(self, text: str) -> str | None:
        upper = text.upper()

        if re.search(r"\bNAM\b", upper):
            return "Nam"

        if "NU" in upper or "NỮ" in upper:
            return "Nữ"

        return None