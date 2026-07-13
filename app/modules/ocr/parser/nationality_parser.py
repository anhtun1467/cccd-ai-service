class NationalityParser:
    """
    Tách quốc tịch.
    """

    def parse(self, text: str) -> str | None:
        upper = text.upper()

        if (
            "VIET" in upper
            or "VICT" in upper
            or "NANA" in upper
        ):
            return "Viet Nam"

        return None