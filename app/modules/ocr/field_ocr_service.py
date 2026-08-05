from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

from app.modules.ocr.base_engine import BaseOCREngine
from app.modules.ocr.easyocr_engine import EasyOCREngine
from app.modules.ocr.field_cropper import CCCDFieldCropper
from app.modules.ocr.text_normalizer import OCRTextNormalizer


class FieldOCRService:
    """
    OCR tß╗½ng v├╣ng th├┤ng tin tr├¬n mß║╖t tr╞░ß╗¢c CCCD.

    Quy tr├¼nh:
        Card image
        -> Chuß║⌐n h├│a ß║únh CCCD
        -> Cß║»t tß╗½ng tr╞░ß╗¥ng
        -> Ph├ít hiß╗çn v├á cß║»t ß║únh ch├ón dung
        -> OCR tß╗½ng tr╞░ß╗¥ng chß╗»
        -> Chuß║⌐n h├│a v─ân bß║ún
        -> L├ám sß║ích theo tß╗½ng loß║íi dß╗» liß╗çu
        -> Trß║ú vß╗ü structured data
    """

    OCR_FIELDS: tuple[str, ...] = (
        "idNumber",
        "fullName",
        "dateOfBirth",
        "gender",
        "nationality",
        "placeOfOrigin",
        "placeOfResidence",
        "dateOfExpiry",
    )

    DATE_PATTERN = re.compile(
        r"(?<!\d)"
        r"(\d{1,2})"
        r"\s*[/.\-]\s*"
        r"(\d{1,2})"
        r"\s*[/.\-]\s*"
        r"(\d{4})"
        r"(?!\d)"
    )

    FIELD_LABEL_PATTERNS: dict[str, tuple[str, ...]] = {
        "idNumber": (
            r"\bso\b",
            r"\bno\b",
            r"\bnumber\b",
        ),
        "fullName": (
            r"\bho\s*va\s*ten\b",
            r"\bfull\s*name\b",
            r"\bhova\s*ten\b",
        ),
        "dateOfBirth": (
            r"\bngay\s*sinh\b",
            r"\bdate\s*of\s*birth\b",
        ),
        "gender": (
            r"\bgioi\s*tinh\b",
            r"\bsex\b",
        ),
        "nationality": (
            r"\bquoc\s*tich\b",
            r"\bnationality\b",
        ),
        "placeOfOrigin": (
            r"\bque\s*quan\b",
            r"\bplace\s*of\s*origin\b",
        ),
        "placeOfResidence": (
            r"\bnoi\s*thuong\s*tru\b",
            r"\bplace\s*of\s*residence\b",
        ),
        "dateOfExpiry": (
            r"\bco\s*gia\s*tri\s*den\b",
            r"\bdate\s*of\s*expiry\b",
            r"\bexpiry\s*date\b",
        ),
    }

    def __init__(
        self,
        engine: BaseOCREngine | None = None,
    ) -> None:
        self.engine = engine or EasyOCREngine()
        self.cropper = CCCDFieldCropper()

    def extract_fields(
        self,
        card_image_path: str,
        output_dir: str,
    ) -> dict[str, Any]:
        """
        Cß║»t ß║únh CCCD v├á OCR tß╗½ng tr╞░ß╗¥ng ri├¬ng biß╗çt.

        Args:
            card_image_path:
                ─É╞░ß╗¥ng dß║½n ß║únh CCCD ─æ├ú ─æ╞░ß╗úc detect v├á perspective transform.

            output_dir:
                Th╞░ mß╗Ñc l╞░u ß║únh field, ß║únh portrait v├á ß║únh debug.

        Returns:
            Dictionary gß╗ôm:
            - structuredData
            - fieldResults
            - portrait
            - debug
        """

        card_path = Path(card_image_path)

        if not card_path.exists():
            raise FileNotFoundError(
                f"Kh├┤ng t├¼m thß║Ñy ß║únh CCCD: {card_path}"
            )

        field_results = self.cropper.crop_fields_from_path(
            image_path=str(card_path),
            output_dir=output_dir,
        )

        structured_data: dict[str, str | None] = {
            field_name: None
            for field_name in self.OCR_FIELDS
        }

        field_ocr_results: dict[str, dict[str, Any]] = {}

        for field_name in self.OCR_FIELDS:
            field_result = field_results.get(field_name)

            if not field_result:
                field_ocr_results[field_name] = (
                    self.build_empty_field_result(
                        field_name=field_name,
                        message="Kh├┤ng t├¼m thß║Ñy v├╣ng cß║»t",
                    )
                )
                continue

            processed_image_path = field_result.get(
                "imagePath"
            )

            raw_image_path = field_result.get(
                "rawImagePath"
            )

            if not processed_image_path:
                field_ocr_results[field_name] = (
                    self.build_empty_field_result(
                        field_name=field_name,
                        message="Kh├┤ng c├│ ─æ╞░ß╗¥ng dß║½n ß║únh field",
                        raw_image_path=raw_image_path,
                    )
                )
                continue

            processed_path = Path(
                str(processed_image_path)
            )

            if not processed_path.exists():
                field_ocr_results[field_name] = (
                    self.build_empty_field_result(
                        field_name=field_name,
                        message=(
                            "Kh├┤ng t├¼m thß║Ñy ß║únh field: "
                            f"{processed_path}"
                        ),
                        image_path=str(processed_path),
                        raw_image_path=raw_image_path,
                    )
                )
                continue

            try:
                ocr_result = self.engine.recognize(
                    str(processed_path)
                )
            except Exception as error:
                field_ocr_results[field_name] = (
                    self.build_empty_field_result(
                        field_name=field_name,
                        message=f"OCR thß║Ñt bß║íi: {error}",
                        image_path=str(processed_path),
                        raw_image_path=raw_image_path,
                    )
                )
                continue

            normalized_lines = (
                OCRTextNormalizer.normalize_lines(
                    ocr_result.raw_text
                )
            )

            joined_text = self.join_field_text(
                field_name=field_name,
                lines=normalized_lines,
            )

            cleaned_value = self.clean_field_value(
                field_name=field_name,
                value=joined_text,
            )

            average_confidence = (
                self.calculate_average_confidence(
                    ocr_result.text_boxes
                )
            )

            structured_data[field_name] = cleaned_value

            field_ocr_results[field_name] = {
                "fieldName": field_name,
                "success": bool(ocr_result.success),
                "message": ocr_result.message,
                "rawText": list(ocr_result.raw_text),
                "normalizedText": normalized_lines,
                "joinedText": joined_text,
                "value": cleaned_value,
                "averageConfidence": average_confidence,
                "imagePath": str(processed_path),
                "rawImagePath": (
                    str(raw_image_path)
                    if raw_image_path
                    else None
                ),
                "box": field_result.get("box"),
                "rawWidth": field_result.get("rawWidth"),
                "rawHeight": field_result.get("rawHeight"),
                "processedWidth": field_result.get(
                    "processedWidth"
                ),
                "processedHeight": field_result.get(
                    "processedHeight"
                ),
            }

        portrait_result = self.prepare_portrait_result(
            field_results.get("portrait")
        )

        return {
            "structuredData": structured_data,
            "fieldResults": field_ocr_results,
            "portrait": portrait_result,
            "debug": field_results.get("_debug"),
        }

    def prepare_portrait_result(
        self,
        portrait_result: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """
        Chuß║⌐n h├│a dß╗» liß╗çu ß║únh ch├ón dung.

        ß║ónh portrait kh├┤ng ─æ╞░ß╗úc OCR. ß║ónh n├áy ─æ╞░ß╗úc d├╣ng cho module
        Face Verification ß╗ƒ b╞░ß╗¢c tiß║┐p theo.
        """

        if not portrait_result:
            return None

        image_path = portrait_result.get("imagePath")
        raw_image_path = portrait_result.get(
            "rawImagePath"
        )

        image_exists = bool(
            image_path
            and Path(str(image_path)).exists()
        )

        raw_image_exists = bool(
            raw_image_path
            and Path(str(raw_image_path)).exists()
        )

        return {
            "success": image_exists,
            "faceDetected": bool(
                portrait_result.get("faceDetected", False)
            ),
            "imagePath": (
                str(image_path)
                if image_path
                else None
            ),
            "rawImagePath": (
                str(raw_image_path)
                if raw_image_path
                else None
            ),
            "imageExists": image_exists,
            "rawImageExists": raw_image_exists,
            "box": portrait_result.get("box"),
            "rawWidth": portrait_result.get("rawWidth"),
            "rawHeight": portrait_result.get("rawHeight"),
            "processedWidth": portrait_result.get(
                "processedWidth"
            ),
            "processedHeight": portrait_result.get(
                "processedHeight"
            ),
        }

    @staticmethod
    def join_field_text(
        field_name: str,
        lines: list[str],
    ) -> str:
        """
        Gh├⌐p c├íc d├▓ng OCR cß╗ºa mß╗Öt field.

        ─Éß╗ïa chß╗ë ─æ╞░ß╗úc gh├⌐p bß║▒ng dß║Ñu phß║⌐y.
        C├íc tr╞░ß╗¥ng c├▓n lß║íi ─æ╞░ß╗úc gh├⌐p bß║▒ng khoß║úng trß║»ng.
        """

        clean_lines = [
            line.strip()
            for line in lines
            if line and line.strip()
        ]

        if not clean_lines:
            return ""

        if field_name in {
            "placeOfOrigin",
            "placeOfResidence",
        }:
            return ", ".join(clean_lines)

        return " ".join(clean_lines)

    def clean_field_value(
        self,
        field_name: str,
        value: str,
    ) -> str | None:
        """
        L├ám sß║ích gi├í trß╗ï OCR theo loß║íi tr╞░ß╗¥ng.
        """

        if not value:
            return None

        cleaned = OCRTextNormalizer.normalize(value)

        cleaned = self.remove_field_labels(
            field_name=field_name,
            value=cleaned,
        )

        if not cleaned:
            return None

        if field_name == "idNumber":
            return self.clean_id_number(cleaned)

        if field_name == "fullName":
            return self.clean_full_name(cleaned)

        if field_name in {
            "dateOfBirth",
            "dateOfExpiry",
        }:
            return self.clean_date(cleaned)

        if field_name == "gender":
            return self.clean_gender(cleaned)

        if field_name == "nationality":
            return self.clean_nationality(cleaned)

        if field_name in {
            "placeOfOrigin",
            "placeOfResidence",
        }:
            return self.clean_address(cleaned)

        return cleaned or None

    def remove_field_labels(
        self,
        field_name: str,
        value: str,
    ) -> str:
        """
        X├│a c├íc nh├ún tiß║┐ng Viß╗çt v├á tiß║┐ng Anh c├▓n s├│t lß║íi trong v├╣ng OCR.
        """

        patterns = self.FIELD_LABEL_PATTERNS.get(
            field_name,
            (),
        )

        cleaned = value

        for pattern in patterns:
            cleaned = re.sub(
                pattern,
                " ",
                cleaned,
                flags=re.IGNORECASE,
            )

        cleaned = re.sub(
            r"^[\s:/,;.\-]+",
            "",
            cleaned,
        )

        cleaned = re.sub(
            r"\s+",
            " ",
            cleaned,
        ).strip()

        return cleaned

    @staticmethod
    def clean_id_number(
        value: str,
    ) -> str | None:
        """
        L├ám sß║ích sß╗æ CCCD.

        Chß╗ë giß╗» chß╗» sß╗æ v├á y├¬u cß║ºu ─æ├║ng 12 sß╗æ.
        """

        substitutions = str.maketrans(
            {
                "O": "0",
                "o": "0",
                "I": "1",
                "i": "1",
                "L": "1",
                "l": "1",
            }
        )

        normalized = value.translate(substitutions)

        digits = re.sub(
            r"\D",
            "",
            normalized,
        )

        match = re.search(
            r"(?<!\d)(\d{12})(?!\d)",
            digits,
        )

        if match:
            return match.group(1)

        return None

    @staticmethod
    def clean_full_name(
        value: str,
    ) -> str | None:
        """
        L├ám sß║ích hß╗ì t├¬n.

        Giß╗» chß╗» c├íi tiß║┐ng Viß╗çt v├á khoß║úng trß║»ng.
        """

        normalized = unicodedata.normalize("NFC", str(value))
        words = re.findall(
            r"[^\W\d_]+",
            normalized,
            flags=re.UNICODE,
        )
        cleaned = " ".join(words).strip()

        if len(cleaned) < 3:
            return None

        words = cleaned.split()

        if len(words) < 2:
            return None

        return cleaned.upper()

    def clean_date(
        self,
        value: str,
    ) -> str | None:
        """
        Chuß║⌐n h├│a ng├áy vß╗ü DD/MM/YYYY v├á kiß╗âm tra ng├áy hß╗úp lß╗ç.
        """

        normalized = value.translate(
            str.maketrans(
                {
                    "O": "0",
                    "o": "0",
                    "I": "1",
                    "i": "1",
                    "l": "1",
                    "h": "/",
                }
            )
        )

        match = self.DATE_PATTERN.search(
            normalized
        )

        if not match:
            digits = re.sub(
                r"\D",
                "",
                normalized,
            )

            if len(digits) == 8:
                normalized = (
                    f"{digits[0:2]}/"
                    f"{digits[2:4]}/"
                    f"{digits[4:8]}"
                )

                match = self.DATE_PATTERN.search(
                    normalized
                )

        if not match:
            return None

        day = int(match.group(1))
        month = int(match.group(2))
        year = int(match.group(3))

        try:
            parsed_date = datetime(
                year=year,
                month=month,
                day=day,
            )
        except ValueError:
            return None

        return parsed_date.strftime("%d/%m/%Y")

    @staticmethod
    def clean_gender(
        value: str,
    ) -> str | None:
        """
        Chuß║⌐n h├│a giß╗¢i t├¡nh vß╗ü Nam hoß║╖c Nß╗».
        """

        normalized = unicodedata.normalize("NFD", str(value))
        plain = "".join(
            character
            for character in normalized
            if unicodedata.category(character) != "Mn"
        )
        plain = plain.replace("đ", "d").replace("Đ", "D")
        compact = re.sub(r"[^a-z]", "", plain.lower())

        if compact in {"nam", "male"}:
            return "Nam"

        if compact in {"nu", "female", "ni", "nw", "nv", "nii"}:
            return "Nữ"

        return None

    @staticmethod
    def clean_nationality(
        value: str,
    ) -> str | None:
        """
        Chuß║⌐n h├│a quß╗æc tß╗ïch.
        """

        normalized = unicodedata.normalize("NFD", str(value))
        plain = "".join(
            character
            for character in normalized
            if unicodedata.category(character) != "Mn"
        )
        lowered = plain.replace("đ", "d").replace("Đ", "D").lower()

        vietnam_patterns = (
            r"\bviet\s*nam\b",
            r"\bvietnam\b",
            r"\bvict\s*nam\b",
            r"\bviet\s*nana\b",
            r"\bvict\s*nana\b",
        )

        for pattern in vietnam_patterns:
            if re.search(pattern, lowered):
                return "Việt Nam"

        cleaned = " ".join(
            re.findall(
                r"[^\W\d_]+",
                unicodedata.normalize("NFC", str(value)),
                flags=re.UNICODE,
            )
        ).strip()

        return cleaned or None

    @staticmethod
    def clean_address(
        value: str,
    ) -> str | None:
        """
        L├ám sß║ích qu├¬ qu├ín hoß║╖c n╞íi th╞░ß╗¥ng tr├║.
        """

        cleaned = value.replace(";", ",")

        cleaned = re.sub(
            r"\s+([,.:])",
            r"\1",
            cleaned,
        )

        cleaned = re.sub(
            r"[,]\s*[,]+",
            ", ",
            cleaned,
        )

        cleaned = re.sub(
            r"\s*,\s*",
            ", ",
            cleaned,
        )

        cleaned = re.sub(
            r"\s+",
            " ",
            cleaned,
        )

        cleaned = cleaned.strip(
            " ,.;:/-"
        )

        if len(cleaned) < 3:
            return None

        return cleaned

    @staticmethod
    def calculate_average_confidence(
        text_boxes: list[Any],
    ) -> float:
        """
        T├¡nh confidence trung b├¼nh cß╗ºa kß║┐t quß║ú OCR.
        """

        if not text_boxes:
            return 0.0

        scores: list[float] = []

        for item in text_boxes:
            confidence = getattr(
                item,
                "confidence",
                None,
            )

            if confidence is None and isinstance(
                item,
                dict,
            ):
                confidence = item.get(
                    "confidence",
                    0.0,
                )

            try:
                score = float(confidence)
            except (TypeError, ValueError):
                continue

            if 0.0 <= score <= 1.0:
                scores.append(score)

        if not scores:
            return 0.0

        return round(
            sum(scores) / len(scores),
            4,
        )

    @staticmethod
    def build_empty_field_result(
        field_name: str,
        message: str,
        image_path: str | None = None,
        raw_image_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Tß║ío kß║┐t quß║ú mß║╖c ─æß╗ïnh khi mß╗Öt field kh├┤ng OCR ─æ╞░ß╗úc.
        """

        return {
            "fieldName": field_name,
            "success": False,
            "message": message,
            "rawText": [],
            "normalizedText": [],
            "joinedText": "",
            "value": None,
            "averageConfidence": 0.0,
            "imagePath": image_path,
            "rawImagePath": raw_image_path,
            "box": None,
        }


field_ocr_service = FieldOCRService()
