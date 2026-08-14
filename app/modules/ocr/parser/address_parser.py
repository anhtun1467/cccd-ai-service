class AddressParser:
    """
    Tách quê quán và nơi thường trú từ OCR lines.
    Cập nhật chuẩn hóa chuỗi và từ khóa dừng.
    """

    ORIGIN_KEYWORDS = [
        "QUE QUAN",
        "PLACE OF ORIGIN",
    ]

    RESIDENCE_KEYWORDS = [
        "NOI THUONG TRU",
        "PLACE OF RESIDENCE",
    ]

    STOP_KEYWORDS = [
        "DATE OF EXPIRY",
        "DATE OFDXPIRY",
        "DANG",
        "TNENG",
        "ISSUE",
        "MRZ",
        "NOI CAP",  # Bổ sung thêm từ khóa thường gặp
        "QUYEN",    # Bổ sung thêm từ khóa thường gặp
    ]

    def parse_place_of_origin(self, lines: list[str]) -> str | None:
        start_index = self.find_keyword_index(lines, self.ORIGIN_KEYWORDS)

        if start_index is None:
            return None

        address_parts = self.collect_address_lines(
            lines=lines,
            start_index=start_index + 1,
            max_lines=4,
        )

        return self.normalize_address(address_parts)

    def parse_place_of_residence(self, lines: list[str]) -> str | None:
        start_index = self.find_keyword_index(lines, self.RESIDENCE_KEYWORDS)

        if start_index is None:
            return None

        first_line = lines[start_index]

        address_parts = []

        if ":" in first_line:
            after_colon = first_line.split(":", 1)[1].strip()
            if after_colon:
                address_parts.append(after_colon)

        address_parts.extend(
            self.collect_address_lines(
                lines=lines,
                start_index=start_index + 1,
                max_lines=3,
            )
        )

        return self.normalize_address(address_parts)

    def find_keyword_index(
        self,
        lines: list[str],
        keywords: list[str],
    ) -> int | None:
        for index, line in enumerate(lines):
            upper_line = line.upper()

            for keyword in keywords:
                if keyword in upper_line:
                    return index

        return None

    def collect_address_lines(
        self,
        lines: list[str],
        start_index: int,
        max_lines: int,
    ) -> list[str]:
        result = []

        for line in lines[start_index : start_index + max_lines]:
            upper_line = line.upper()

            if self.is_stop_line(upper_line):
                break

            if self.is_noise_line(upper_line):
                continue

            result.append(line)

        return result

    def is_stop_line(self, upper_line: str) -> bool:
        return any(keyword in upper_line for keyword in self.STOP_KEYWORDS)

    def is_noise_line(self, upper_line: str) -> bool:
        if not upper_line:
            return True

        if upper_line in {"E"}:
            return True

        return False

    def normalize_address(self, parts: list[str]) -> str | None:
        if not parts:
            return None

        cleaned_parts = []

        for part in parts:
            part = part.strip()
            # Cải tiến và bổ sung các lỗi OCR thường gặp
            part = part.replace("Ha Noii", "Ha Noi")
            part = part.replace("Ha No", "Ha Noi")
            part = part.replace("Thanh Zuah", "Thanh Xuan")
            part = part.replace("Neech", "Ngach")
            part = part.replace("  ", " ")
            # Thêm xử lý dọn dẹp ký tự thừa nếu có
            part = part.strip(" ,.-")

            if part:
                cleaned_parts.append(part)

        if not cleaned_parts:
            return None

        return ", ".join(cleaned_parts)