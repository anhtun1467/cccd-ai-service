import re


class BirthParser:
    """
    Tách ngày sinh từ chuỗi văn bản OCR.
    Hỗ trợ nhận diện linh hoạt các định dạng ngày/tháng và chuẩn hóa về DD/MM/YYYY.
    """

    def parse(self, text: str) -> str | None:
        if not text:
            return None

        # Cải tiến regex: Cho phép dấu / hoặc -, hỗ trợ cả 1 hoặc 2 chữ số cho ngày/tháng
        match = re.search(
            r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})\b",
            text,
        )

        if match:
            day, month, year = match.groups()
            # Chuẩn hóa về dạng chuẩn DD/MM/YYYY
            return f"{int(day):02d}/{int(month):02d}/{year}"

        return None