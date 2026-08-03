from __future__ import annotations

import re
import unicodedata


class OCRTextNormalizer:
    """
    Chuẩn hóa văn bản OCR trên CCCD.

    Chức năng:
    - Chuẩn hóa Unicode.
    - Xóa khoảng trắng dư.
    - Thay dấu gạch dưới bằng khoảng trắng.
    - Chuẩn hóa dấu "/" trong ngày tháng và nhãn song ngữ.
    - Sửa các nhãn CCCD thường bị OCR nhận dính chữ.
    - Sửa một số lỗi OCR phổ biến.
    """

    PHRASE_REPLACEMENTS: tuple[tuple[str, str], ...] = (
        # Nhãn tiếng Việt
        (r"\bhova\s*ten\b", "Ho va ten"),
        (r"\bhovaten\b", "Ho va ten"),
        (r"\bho\s*va\s*ten\b", "Ho va ten"),

        (r"\bngaysinh\b", "Ngay sinh"),
        (r"\bngay\s*sinh\b", "Ngay sinh"),

        (r"\bgioitinh\b", "Gioi tinh"),
        (r"\bgioi\s*tinh\b", "Gioi tinh"),

        (r"\bquoctich\b", "Quoc tich"),
        (r"\bquoc\s*tich\b", "Quoc tich"),

        (r"\bquequan\b", "Que quan"),
        (r"\bque\s*quan\b", "Que quan"),

        (r"\bnoithuongtru\b", "Noi thuong tru"),
        (r"\bnoi\s*thuong\s*tru\b", "Noi thuong tru"),

        (r"\bcancuoccongdan\b", "CAN CUOC CONG DAN"),
        (
            r"\bcan\s*cuoc\s*conc\s*dan\b",
            "CAN CUOC CONG DAN",
        ),

        # Nhãn tiếng Anh
        (r"\bfullname\b", "Full name"),

        (r"\bdateofbirth\b", "Date of birth"),
        (r"\bdate\s*ofbirth\b", "Date of birth"),
        (r"\bdateof\s*birth\b", "Date of birth"),

        (r"\bplaceoforigin\b", "Place of origin"),
        (r"\bplace\s*oforigin\b", "Place of origin"),
        (r"\bplaceof\s*origin\b", "Place of origin"),

        (
            r"\bplaceofresidence\b",
            "Place of residence",
        ),
        (
            r"\bplace\s*ofresidence\b",
            "Place of residence",
        ),
        (
            r"\bplaceof\s*residence\b",
            "Place of residence",
        ),

        (r"\bdateofexpiry\b", "Date of Expiry"),
        (
            r"\bdate\s*ofd[a-z]*piry\b",
            "Date of Expiry",
        ),
        (
            r"\bdate\s*of\s*[dexp]{1,3}piry\b",
            "Date of Expiry",
        ),

        (
            r"\bcitizenidentitycard\b",
            "Citizen Identity Card",
        ),

        # Quốc tịch thường bị OCR sai
        (r"\bvict\s*nana\b", "Viet Nam"),
        (r"\bviet\s*nana\b", "Viet Nam"),
        (r"\bvict\s*nam\b", "Viet Nam"),
        (r"\bvietnam\b", "Viet Nam"),

        # Cụm từ cố định
        (
            r"\bfreedom\s*happiness\b",
            "Freedom - Happiness",
        ),
        (
            r"\bindependence\s*freedom\b",
            "Independence - Freedom",
        ),
    )

    @staticmethod
    def _remove_accents(value: str) -> str:
        """Tạo bản không dấu để so khớp nhưng không sửa dữ liệu gốc."""
        normalized = unicodedata.normalize("NFD", value)
        plain = "".join(
            character
            for character in normalized
            if unicodedata.category(character) != "Mn"
        )
        return plain.replace("đ", "d").replace("Đ", "D")

    @classmethod
    def _replace_phrase(
        cls,
        value: str,
        pattern: str,
        replacement: str,
    ) -> str:
        """
        Chuẩn hóa nhãn không phân biệt dấu tiếng Việt.

        Chỉ nhãn được thay thế; phần dữ liệu như họ tên và địa chỉ vẫn
        giữ nguyên dấu mà OCR tiếng Việt đã nhận được.
        """
        plain_value = cls._remove_accents(value)
        matches = list(
            re.finditer(
                pattern,
                plain_value,
                flags=re.IGNORECASE,
            )
        )

        for match in reversed(matches):
            value = (
                value[:match.start()]
                + replacement
                + value[match.end():]
            )

        return value

    @classmethod
    def normalize(
        cls,
        text: str | None,
    ) -> str:
        """
        Chuẩn hóa một chuỗi OCR.

        Args:
            text: Văn bản OCR đầu vào.

        Returns:
            Văn bản đã được chuẩn hóa.
        """

        if not text:
            return ""

        value = unicodedata.normalize(
            "NFKC",
            str(text),
        )

        # Chuẩn hóa các ký tự phân cách.
        value = value.replace("_", " ")
        value = value.replace("|", "/")
        value = value.replace("–", "-")
        value = value.replace("—", "-")

        # Loại bỏ ký tự điều khiển.
        value = "".join(
            char
            for char in value
            if unicodedata.category(char)[0] != "C"
        )

        # Thu gọn khoảng trắng ban đầu.
        value = re.sub(
            r"\s+",
            " ",
            value,
        ).strip()

        # Xóa khoảng trắng trước dấu câu.
        # Không xử lý dấu "/" tại bước này.
        value = re.sub(
            r"\s+([,.:;])",
            r"\1",
            value,
        )

        # Thêm khoảng trắng sau dấu câu khi bị dính chữ.
        value = re.sub(
            r"([,:;])(?=[A-Za-zÀ-ỹ0-9])",
            r"\1 ",
            value,
        )

        # Chuẩn hóa ngày tháng:
        # 24 / 03 / 1995 -> 24/03/1995
        value = re.sub(
            r"(?<=\d)\s*/\s*(?=\d)",
            "/",
            value,
        )

        # Chuẩn hóa dấu "/" giữa hai phần chữ:
        # Ho va ten/ Full name -> Ho va ten / Full name
        value = re.sub(
            r"(?<=[A-Za-zÀ-ỹ])\s*/\s*(?=[A-Za-zÀ-ỹ])",
            " / ",
            value,
        )

        # Trường hợp trước "/" là chữ,
        # sau "/" có khoảng trắng rồi mới tới chữ.
        value = re.sub(
            r"(?<=[A-Za-zÀ-ỹ])\s*/\s+(?=[A-Za-zÀ-ỹ])",
            " / ",
            value,
        )

        # Sửa các cụm từ OCR bị dính hoặc nhận sai.
        for pattern, replacement in cls.PHRASE_REPLACEMENTS:
            value = cls._replace_phrase(
                value,
                pattern,
                replacement,
            )

        # Chạy lại chuẩn hóa ngày tháng sau khi thay thế.
        value = re.sub(
            r"(?<=\d)\s*/\s*(?=\d)",
            "/",
            value,
        )

        # Chạy lại chuẩn hóa nhãn song ngữ.
        value = re.sub(
            r"(?<=[A-Za-zÀ-ỹ])\s*/\s*(?=[A-Za-zÀ-ỹ])",
            " / ",
            value,
        )

        # Xóa khoảng trắng dư lần cuối.
        value = re.sub(
            r"\s+",
            " ",
            value,
        ).strip()

        return value

    @classmethod
    def normalize_lines(
        cls,
        lines: list[str],
    ) -> list[str]:
        """
        Chuẩn hóa danh sách các dòng OCR.

        Args:
            lines: Danh sách văn bản OCR.

        Returns:
            Danh sách văn bản đã chuẩn hóa và loại bỏ dòng rỗng.
        """

        normalized_lines: list[str] = []

        for line in lines:
            clean_line = cls.normalize(line)

            if clean_line:
                normalized_lines.append(clean_line)

        return normalized_lines