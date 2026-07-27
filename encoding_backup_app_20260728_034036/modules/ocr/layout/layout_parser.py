from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from app.modules.ocr.layout.anchor_finder import (
    AnchorFinder,
    CandidateMatch,
)
from app.modules.ocr.layout.box_utils import (
    prepare_layout_boxes,
)
from app.modules.ocr.layout.text_box import (
    LayoutTextBox,
)


ID_ALIASES = (
    "Số",
    "So",
    "Số định danh cá nhân",
    "So dinh danh ca nhan",
    "No",
    "No.",
)

FULL_NAME_ALIASES = (
    "Họ và tên",
    "Ho va ten",
    "Họ tên",
    "Ho ten",
    "Full name",
)

DATE_OF_BIRTH_ALIASES = (
    "Ngày sinh",
    "Ngay sinh",
    "Date of birth",
    "Birth date",
)

GENDER_ALIASES = (
    "Giới tính",
    "Gioi tinh",
    "Sex",
    "Gender",
)

NATIONALITY_ALIASES = (
    "Quốc tịch",
    "Quoc tich",
    "Nationality",
)

PLACE_OF_ORIGIN_ALIASES = (
    "Quê quán",
    "Que quan",
    "Place of origin",
    "Place of birth",
)

PLACE_OF_RESIDENCE_ALIASES = (
    "Nơi thường trú",
    "Noi thuong tru",
    "Nơi cư trú",
    "Noi cu tru",
    "Place of residence",
    "Permanent residence",
)

DATE_OF_EXPIRY_ALIASES = (
    "Có giá trị đến",
    "Co gia tri den",
    "Ngày hết hạn",
    "Ngay het han",
    "Date of expiry",
    "Valid until",
)


@dataclass(slots=True)
class CCCDLayoutResult:
    """
    Kết quả trích xuất thông tin CCCD bằng Layout Parser.
    """

    id_number: str = ""
    full_name: str = ""
    date_of_birth: str = ""
    gender: str = ""
    nationality: str = ""
    place_of_origin: str = ""
    place_of_residence: str = ""
    date_of_expiry: str = ""

    def to_dict(
        self,
        camel_case: bool = False,
    ) -> dict[str, str]:
        """
        Chuyển kết quả thành dictionary.

        camel_case=False:
            {
                "id_number": "...",
                "full_name": "..."
            }

        camel_case=True:
            {
                "idNumber": "...",
                "fullName": "..."
            }
        """

        if not camel_case:
            return asdict(self)

        return {
            "idNumber": self.id_number,
            "fullName": self.full_name,
            "dateOfBirth": self.date_of_birth,
            "gender": self.gender,
            "nationality": self.nationality,
            "placeOfOrigin": self.place_of_origin,
            "placeOfResidence": self.place_of_residence,
            "dateOfExpiry": self.date_of_expiry,
        }


@dataclass(slots=True)
class CCCDLayoutDebugResult:
    """
    Kết quả debug giúp kiểm tra anchor và candidate đã được chọn.
    """

    result: CCCDLayoutResult
    matches: dict[str, dict | None]

    def to_dict(self) -> dict:
        return {
            "result": self.result.to_dict(
                camel_case=True
            ),
            "matches": self.matches,
        }


class CCCDLayoutParser:
    """
    Parser CCCD dựa trên vị trí và quan hệ hình học
    giữa các OCR text box.

    Luồng xử lý:

        OCR text boxes
            ↓
        chuẩn hóa LayoutTextBox
            ↓
        AnchorFinder
            ↓
        chọn candidate bên phải hoặc phía dưới
            ↓
        làm sạch dữ liệu
            ↓
        CCCDLayoutResult
    """

    def __init__(
        self,
        boxes: Sequence[LayoutTextBox] | Iterable[object],
        minimum_confidence: float = 0.05,
        minimum_anchor_similarity: float = 0.60,
    ) -> None:
        self.boxes = prepare_layout_boxes(
            items=boxes,
            minimum_confidence=minimum_confidence,
            remove_duplicates=True,
            merge_lines=False,
        )

        self.finder = AnchorFinder(
            boxes=self.boxes,
            minimum_anchor_similarity=(
                minimum_anchor_similarity
            ),
        )

        self._matches: dict[
            str,
            CandidateMatch | None
        ] = {}

    def parse(self) -> CCCDLayoutResult:
        """
        Trích xuất toàn bộ thông tin CCCD.
        """

        self._matches.clear()

        return CCCDLayoutResult(
            id_number=self.parse_id_number(),
            full_name=self.parse_full_name(),
            date_of_birth=(
                self.parse_date_of_birth()
            ),
            gender=self.parse_gender(),
            nationality=self.parse_nationality(),
            place_of_origin=(
                self.parse_place_of_origin()
            ),
            place_of_residence=(
                self.parse_place_of_residence()
            ),
            date_of_expiry=(
                self.parse_date_of_expiry()
            ),
        )

    def parse_with_debug(
        self,
    ) -> CCCDLayoutDebugResult:
        """
        Trả về kết quả cùng thông tin anchor/candidate để debug.
        """

        result = self.parse()

        debug_matches: dict[
            str,
            dict | None
        ] = {}

        for field_name, match in (
            self._matches.items()
        ):
            debug_matches[field_name] = (
                match.to_dict()
                if match is not None
                else None
            )

        return CCCDLayoutDebugResult(
            result=result,
            matches=debug_matches,
        )

    def parse_id_number(self) -> str:
        match = self._find_field(
            field_name="idNumber",
            aliases=ID_ALIASES,
            preferred_direction="right",
            fallback_direction="below",
            minimum_similarity=0.55,
        )

        if match is None:
            return self._find_id_number_fallback()

        cleaned = self.clean_id_number(
            match.candidate.text
        )

        if cleaned:
            return cleaned

        return self._find_id_number_fallback()

    def parse_full_name(self) -> str:
        match = self._find_field(
            field_name="fullName",
            aliases=FULL_NAME_ALIASES,
            preferred_direction="below",
            fallback_direction="right",
        )

        if match is None:
            return ""

        return self.clean_full_name(
            match.candidate.text
        )

    def parse_date_of_birth(self) -> str:
        match = self._find_field(
            field_name="dateOfBirth",
            aliases=DATE_OF_BIRTH_ALIASES,
            preferred_direction="right",
            fallback_direction="below",
        )

        if match is None:
            return self._find_date_fallback(
                exclude_expiry_area=True
            )

        cleaned = self.clean_date(
            match.candidate.text
        )

        if cleaned:
            return cleaned

        return self._find_date_fallback(
            exclude_expiry_area=True
        )

    def parse_gender(self) -> str:
        match = self._find_field(
            field_name="gender",
            aliases=GENDER_ALIASES,
            preferred_direction="right",
            fallback_direction="below",
        )

        if match is None:
            return ""

        return self.clean_gender(
            match.candidate.text
        )

    def parse_nationality(self) -> str:
        match = self._find_field(
            field_name="nationality",
            aliases=NATIONALITY_ALIASES,
            preferred_direction="right",
            fallback_direction="below",
        )

        if match is None:
            return ""

        return self.clean_nationality(
            match.candidate.text
        )

    def parse_place_of_origin(self) -> str:
        match = self._find_field(
            field_name="placeOfOrigin",
            aliases=PLACE_OF_ORIGIN_ALIASES,
            preferred_direction="below",
            fallback_direction="right",
        )

        if match is None:
            return ""

        return self.clean_address(
            match.candidate.text
        )

    def parse_place_of_residence(self) -> str:
        match = self._find_field(
            field_name="placeOfResidence",
            aliases=PLACE_OF_RESIDENCE_ALIASES,
            preferred_direction="below",
            fallback_direction="right",
        )

        if match is None:
            return ""

        return self.clean_address(
            match.candidate.text
        )

    def parse_date_of_expiry(self) -> str:
        match = self._find_field(
            field_name="dateOfExpiry",
            aliases=DATE_OF_EXPIRY_ALIASES,
            preferred_direction="right",
            fallback_direction="below",
        )

        if match is None:
            return ""

        return self.clean_date(
            match.candidate.text
        )

    def _find_field(
        self,
        field_name: str,
        aliases: Iterable[str],
        preferred_direction: str,
        fallback_direction: str | None,
        minimum_similarity: float | None = None,
    ) -> CandidateMatch | None:
        """
        Tìm giá trị của một trường và lưu lại thông tin debug.
        """

        excluded_boxes = self._get_used_boxes()

        match = self.finder.find_value(
            anchor_name=field_name,
            aliases=aliases,
            preferred_direction=(
                preferred_direction
            ),
            fallback_direction=(
                fallback_direction
            ),
            minimum_similarity=(
                minimum_similarity
            ),
            excluded_boxes=excluded_boxes,
        )

        self._matches[field_name] = match

        return match

    def _get_used_boxes(
        self,
    ) -> list[LayoutTextBox]:
        """
        Trả về các candidate đã được dùng để tránh một box
        bị gán cho nhiều trường khác nhau.
        """

        used_boxes: list[
            LayoutTextBox
        ] = []

        for match in self._matches.values():
            if match is None:
                continue

            used_boxes.append(
                match.candidate
            )

        return used_boxes

    def _find_id_number_fallback(
        self,
    ) -> str:
        """
        Fallback khi không tìm thấy nhãn số CCCD.

        Chọn chuỗi gồm đúng 12 chữ số có confidence cao nhất.
        """

        candidates: list[
            tuple[float, str]
        ] = []

        for box in self.boxes:
            value = self.clean_id_number(
                box.text
            )

            if len(value) != 12:
                continue

            score = (
                box.confidence * 2.0
                - box.center_y * 0.0001
            )

            candidates.append(
                (score, value)
            )

        if not candidates:
            return ""

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return candidates[0][1]

    def _find_date_fallback(
        self,
        exclude_expiry_area: bool = False,
    ) -> str:
        """
        Tìm ngày tháng trong toàn bộ OCR boxes.

        Ưu tiên ngày nằm ở nửa trên của thẻ,
        phù hợp với vị trí ngày sinh.
        """

        candidates: list[
            tuple[float, str]
        ] = []

        if not self.boxes:
            return ""

        maximum_bottom = max(
            box.bottom
            for box in self.boxes
        )

        for box in self.boxes:
            date_value = self.clean_date(
                box.text
            )

            if not date_value:
                continue

            relative_y = (
                box.center_y
                / maximum_bottom
                if maximum_bottom > 0
                else 1.0
            )

            if (
                exclude_expiry_area
                and relative_y > 0.80
            ):
                continue

            score = (
                box.confidence * 2.0
                - relative_y
            )

            candidates.append(
                (
                    score,
                    date_value,
                )
            )

        if not candidates:
            return ""

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return candidates[0][1]

    @staticmethod
    def clean_id_number(
        text: str,
    ) -> str:
        """
        Chuẩn hóa số CCCD.

        Sửa một số ký tự OCR thường nhầm:
        O -> 0
        I, l -> 1
        S -> 5
        B -> 8
        """

        normalized = (
            text.upper()
            .replace("O", "0")
            .replace("I", "1")
            .replace("L", "1")
            .replace("S", "5")
            .replace("B", "8")
        )

        digits = re.sub(
            r"\D",
            "",
            normalized,
        )

        match = re.search(
            r"\d{12}",
            digits,
        )

        if match:
            return match.group(0)

        return digits

    @staticmethod
    def clean_full_name(
        text: str,
    ) -> str:
        """
        Chuẩn hóa họ tên.
        """

        value = CCCDLayoutParser._remove_label_prefixes(
            text=text,
            prefixes=FULL_NAME_ALIASES,
        )

        value = re.sub(
            r"[^A-Za-zÀ-ỹĐđ\s'-]",
            " ",
            value,
        )

        value = CCCDLayoutParser._normalize_spaces(
            value
        )

        return value.strip(
            " :-/"
        )

    @staticmethod
    def clean_date(
        text: str,
    ) -> str:
        """
        Chuẩn hóa ngày về dạng DD/MM/YYYY.
        """

        value = (
            text.strip()
            .replace(".", "/")
            .replace("-", "/")
            .replace("\\", "/")
        )

        value = re.sub(
            r"\s*/\s*",
            "/",
            value,
        )

        match = re.search(
            r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b",
            value,
        )

        if match is None:
            compact_digits = re.sub(
                r"\D",
                "",
                value,
            )

            if len(compact_digits) == 8:
                day = compact_digits[:2]
                month = compact_digits[2:4]
                year = compact_digits[4:]

                return (
                    f"{day}/{month}/{year}"
                )

            return ""

        day = int(
            match.group(1)
        )

        month = int(
            match.group(2)
        )

        year = int(
            match.group(3)
        )

        if not (
            1 <= day <= 31
            and 1 <= month <= 12
            and 1900 <= year <= 2199
        ):
            return ""

        return (
            f"{day:02d}/"
            f"{month:02d}/"
            f"{year:04d}"
        )

    @staticmethod
    def clean_gender(
        text: str,
    ) -> str:
        """
        Chuẩn hóa giới tính về Nam hoặc Nữ.
        """

        value = (
            CCCDLayoutParser
            ._remove_label_prefixes(
                text=text,
                prefixes=GENDER_ALIASES,
            )
            .strip()
            .lower()
        )

        compact = re.sub(
            r"[^a-zà-ỹđ]",
            "",
            value,
        )

        if compact in {
            "nam",
            "male",
        }:
            return "Nam"

        if compact in {
            "nu",
            "nữ",
            "female",
        }:
            return "Nữ"

        if "female" in compact:
            return "Nữ"

        if "male" in compact:
            return "Nam"

        if "nu" in compact:
            return "Nữ"

        if "nam" in compact:
            return "Nam"

        return CCCDLayoutParser._normalize_spaces(
            text
        )

    @staticmethod
    def clean_nationality(
        text: str,
    ) -> str:
        """
        Chuẩn hóa quốc tịch.
        """

        value = CCCDLayoutParser._remove_label_prefixes(
            text=text,
            prefixes=NATIONALITY_ALIASES,
        )

        value = CCCDLayoutParser._normalize_spaces(
            value
        ).strip(
            " :-/"
        )

        compact = re.sub(
            r"[^a-z]",
            "",
            value.lower(),
        )

        vietnam_variants = {
            "vietnam",
            "vietnan",
            "vietnarn",
            "victnam",
            "vietnarn",
        }

        if compact in vietnam_variants:
            return "Việt Nam"

        return value

    @staticmethod
    def clean_address(
        text: str,
    ) -> str:
        """
        Chuẩn hóa quê quán hoặc nơi thường trú.
        """

        all_address_prefixes = (
            PLACE_OF_ORIGIN_ALIASES
            + PLACE_OF_RESIDENCE_ALIASES
        )

        value = CCCDLayoutParser._remove_label_prefixes(
            text=text,
            prefixes=all_address_prefixes,
        )

        value = re.sub(
            r"\s*,\s*",
            ", ",
            value,
        )

        value = re.sub(
            r"\s*;\s*",
            ", ",
            value,
        )

        value = CCCDLayoutParser._normalize_spaces(
            value
        )

        return value.strip(
            " :-/,"
        )

    @staticmethod
    def _remove_label_prefixes(
        text: str,
        prefixes: Iterable[str],
    ) -> str:
        """
        Xóa label nếu OCR gộp label và value trong cùng một box.
        """

        value = text.strip()

        for prefix in prefixes:
            escaped_prefix = re.escape(
                prefix
            )

            value = re.sub(
                pattern=(
                    rf"^\s*{escaped_prefix}"
                    rf"\s*[:/\-]*\s*"
                ),
                repl="",
                string=value,
                flags=re.IGNORECASE,
            )

        return value

    @staticmethod
    def _normalize_spaces(
        text: str,
    ) -> str:
        """
        Xóa khoảng trắng dư.
        """

        return re.sub(
            r"\s+",
            " ",
            text,
        ).strip()
