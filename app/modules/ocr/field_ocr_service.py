from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

from app.modules.ocr.base_engine import BaseOCREngine
from app.modules.ocr.easyocr_engine import EasyOCREngine
from app.modules.ocr.field_cropper import CCCDFieldCropper
from app.modules.ocr.line_merger import OCRLineMerger
from app.modules.ocr.text_normalizer import OCRTextNormalizer


class FieldOCRService:
    """
    OCR từng vùng thông tin trên mặt trước CCCD.

    Quy trình:
        Card image
        -> Chuẩn hóa ảnh CCCD
        -> Cắt từng trường
        -> Phát hiện và cắt ảnh chân dung
        -> OCR từng trường chữ
        -> Chuẩn hóa văn bản
        -> Làm sạch theo từng loại dữ liệu
        -> Trả về structured data
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

    OCR_RETRY_FIELDS = {
        "idNumber",
        "fullName",
        "dateOfBirth",
        "gender",
        "nationality",
        "placeOfOrigin",
        "placeOfResidence",
        "dateOfExpiry",
    }

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
            r"\bho\s*va{1,2}\s*ten\b",
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
            r"\bplace\s*[o0a][fl0]\s*o(?:ri|r|n)?g(?:i)?n\b",
        ),
        "placeOfResidence": (
            r"\bn[o0](?:i|[1l])?\s*thu[o0]ng\s*tr?u\b",
            r"\bplace\s*[o0a][fl0]\s*resi(?:d[eaou]n(?:c[eoa])?)?\b",
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
        # Khởi tạo chậm để pipeline có thể dùng chung một EasyOCR Reader
        # cho OCR toàn thẻ và OCR từng vùng, tránh nạp hai bộ model CPU.
        self.engine = engine
        self.cropper = CCCDFieldCropper()
        self.line_merger = OCRLineMerger(
            vertical_tolerance_ratio=0.30,
            maximum_horizontal_gap_ratio=2.2,
            minimum_vertical_overlap_ratio=0.40,
        )

    def extract_fields(
        self,
        card_image_path: str,
        output_dir: str,
        layout_y_offset: float = 0.0,
    ) -> dict[str, Any]:
        """
        Cắt ảnh CCCD và OCR từng trường riêng biệt.

        Args:
            card_image_path:
                Đường dẫn ảnh CCCD đã được detect và perspective transform.

            output_dir:
                Thư mục lưu ảnh field, ảnh portrait và ảnh debug.

        Returns:
            Dictionary gồm:
            - structuredData
            - fieldResults
            - portrait
            - debug
        """

        card_path = Path(card_image_path)

        if not card_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy ảnh CCCD: {card_path}"
            )

        field_results = self.cropper.crop_fields_from_path(
            image_path=str(card_path),
            output_dir=output_dir,
            layout_y_offset=layout_y_offset,
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
                        message="Không tìm thấy vùng cắt",
                    )
                )
                continue

            processed_image_path = field_result.get("imagePath")
            raw_image_path = field_result.get("rawImagePath")
            variant_specs = field_result.get("variantImagePaths")

            if not isinstance(variant_specs, list):
                variant_specs = []
            if not variant_specs:
                variant_specs = [
                    {
                        "variant": "wide_processed",
                        "imagePath": processed_image_path,
                    },
                    {
                        "variant": "wide_raw",
                        "imagePath": raw_image_path,
                    },
                ]

            candidates: list[dict[str, Any]] = []
            errors: list[str] = []
            visited_paths: set[str] = set()
            maximum_attempts = (
                6
                if field_name == "dateOfExpiry"
                else 5
                if field_name in {
                    "fullName",
                    "placeOfOrigin",
                    "placeOfResidence",
                }
                else 3
                if field_name in self.OCR_RETRY_FIELDS
                else 1
            )

            for spec in variant_specs:
                if len(candidates) >= maximum_attempts:
                    break
                if not isinstance(spec, dict):
                    continue

                image_path_value = spec.get("imagePath")
                variant = str(spec.get("variant") or "unknown")
                if not image_path_value:
                    continue

                candidate_path = Path(str(image_path_value))
                canonical_path = str(candidate_path)
                if canonical_path in visited_paths:
                    continue
                visited_paths.add(canonical_path)
                if not candidate_path.exists():
                    errors.append(
                        f"Không tìm thấy {variant}: {candidate_path}"
                    )
                    continue

                try:
                    candidate = self.recognize_field_candidate(
                        field_name=field_name,
                        image_path=candidate_path,
                        variant=variant,
                    )
                except Exception as error:
                    errors.append(f"{variant}: {error}")
                    continue

                candidates.append(candidate)
                if self.can_stop_field_retries(
                    field_name=field_name,
                    candidates=candidates,
                ):
                    break

            if not candidates:
                field_ocr_results[field_name] = (
                    self.build_empty_field_result(
                        field_name=field_name,
                        message=(
                            "; ".join(errors)
                            if errors
                            else "Không có ảnh field hợp lệ để OCR"
                        ),
                        image_path=(
                            str(processed_image_path)
                            if processed_image_path
                            else None
                        ),
                        raw_image_path=(
                            str(raw_image_path)
                            if raw_image_path
                            else None
                        ),
                    )
                )
                continue

            selected = self.select_best_field_candidate(
                field_name=field_name,
                candidates=candidates,
            )
            cleaned_value = selected.get("value")
            structured_data[field_name] = cleaned_value

            public_candidates = [
                {
                    key: value
                    for key, value in candidate.items()
                    if not key.startswith("_")
                }
                for candidate in candidates
            ]

            field_ocr_results[field_name] = {
                "fieldName": field_name,
                "success": bool(selected.get("success")),
                "message": selected.get("message"),
                "rawText": list(selected.get("rawText") or []),
                "normalizedText": list(
                    selected.get("normalizedText") or []
                ),
                "joinedText": str(selected.get("joinedText") or ""),
                "value": cleaned_value,
                "averageConfidence": float(
                    selected.get("confidence") or 0.0
                ),
                "ocrVariant": selected.get("variant"),
                "ocrCandidates": public_candidates,
                "imagePath": selected.get("imagePath"),
                "rawImagePath": (
                    str(raw_image_path)
                    if raw_image_path
                    else None
                ),
                "box": field_result.get("box"),
                "rawWidth": field_result.get("rawWidth"),
                "rawHeight": field_result.get("rawHeight"),
                "processedWidth": field_result.get("processedWidth"),
                "processedHeight": field_result.get("processedHeight"),
                "attemptCount": len(candidates),
                "retryErrors": errors,
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

    def extract_normalized_lines(
        self,
        ocr_result: Any,
    ) -> list[str]:
        """Ghép text box theo tọa độ trước khi nối giá trị của field."""
        text_boxes: list[dict[str, Any]] = []
        for item in getattr(ocr_result, "text_boxes", []) or []:
            text = str(getattr(item, "text", "")).strip()
            box = getattr(item, "box", None)
            if not text or not box:
                continue
            text_boxes.append(
                {
                    "text": text,
                    "confidence": float(
                        getattr(item, "confidence", 0.0)
                    ),
                    "box": box,
                }
            )

        merged = self.line_merger.merge(text_boxes)
        if merged:
            return OCRTextNormalizer.normalize_lines(
                [str(item.get("text", "")) for item in merged]
            )
        return OCRTextNormalizer.normalize_lines(
            list(getattr(ocr_result, "raw_text", []) or [])
        )

    def recognize_field_candidate(
        self,
        field_name: str,
        image_path: Path,
        variant: str,
    ) -> dict[str, Any]:
        """OCR và chuẩn hóa một biến thể ảnh của một trường."""
        if self.engine is None:
            self.engine = EasyOCREngine()
        field_recognizer = getattr(self.engine, "recognize_field", None)
        if callable(field_recognizer):
            ocr_result = field_recognizer(
                str(image_path),
                field_name,
            )
        else:
            ocr_result = self.engine.recognize(str(image_path))

        normalized_lines = self.extract_normalized_lines(ocr_result)
        joined_text = self.join_field_text(
            field_name=field_name,
            lines=normalized_lines,
        )
        cleaned_value = self.clean_field_value(
            field_name=field_name,
            value=joined_text,
        )
        confidence = self.calculate_average_confidence(
            list(getattr(ocr_result, "text_boxes", []) or [])
        )

        return {
            "variant": variant,
            "value": cleaned_value,
            "confidence": confidence,
            "normalizedText": normalized_lines,
            "joinedText": joined_text,
            "rawText": list(getattr(ocr_result, "raw_text", []) or []),
            "success": bool(getattr(ocr_result, "success", False)),
            "message": str(getattr(ocr_result, "message", "")),
            "imagePath": str(image_path),
        }

    @classmethod
    def can_stop_field_retries(
        cls,
        field_name: str,
        candidates: list[dict[str, Any]],
    ) -> bool:
        """Dừng sớm khi đã có giá trị đủ mạnh, tránh tăng thời gian CPU."""
        if not candidates:
            return False

        candidate = candidates[-1]
        value = candidate.get("value")
        if not value:
            return False
        text = str(value).strip()

        if field_name == "idNumber":
            return bool(re.fullmatch(r"\d{12}", text))

        if field_name == "dateOfExpiry":
            try:
                parsed = datetime.strptime(text, "%d/%m/%Y")
            except ValueError:
                return False
            if not 1900 <= parsed.year <= 2100:
                return False
            key = cls.field_candidate_key(field_name, text)
            support = sum(
                cls.field_candidate_key(
                    field_name,
                    item.get("value"),
                ) == key
                for item in candidates
            )
            return support >= 2

        if field_name == "dateOfBirth":
            try:
                parsed = datetime.strptime(text, "%d/%m/%Y")
            except ValueError:
                return False
            return 1900 <= parsed.year <= 2100

        if field_name == "gender":
            return text in {"Nam", "Nữ"}

        if field_name == "nationality":
            return text == "Việt Nam"

        if field_name == "fullName":
            valid_name = 2 <= len(text.split()) <= 7
            key = cls.field_candidate_key(field_name, text)
            support = sum(
                cls.field_candidate_key(
                    field_name,
                    item.get("value"),
                ) == key
                for item in candidates
            )
            return bool(
                valid_name
                and len(candidates) >= 2
                and support >= 2
                and cls.diacritic_score(text) > 0
            )

        if field_name in {"placeOfOrigin", "placeOfResidence"}:
            components = [
                item.strip()
                for item in re.split(r"[,;]", text)
                if item.strip()
            ]
            key = cls.field_candidate_key(field_name, text)
            support = sum(
                cls.field_candidate_key(
                    field_name,
                    item.get("value"),
                ) == key
                for item in candidates
            )
            return bool(
                len(candidates) >= 2
                and support >= 2
                and len(text.split()) >= 3
                and len(components) >= 2
                and cls.diacritic_score(text) > 0
            )

        return True

    @classmethod
    def select_best_field_candidate(
        cls,
        field_name: str,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Chọn ứng viên theo kiểu trường, đồng thuận và lượng dấu giữ được."""
        available = [candidate for candidate in candidates if candidate.get("value")]
        if not available:
            return max(
                candidates,
                key=lambda item: float(item.get("confidence") or 0.0),
            )

        keys = [
            cls.field_candidate_key(field_name, candidate.get("value"))
            for candidate in available
        ]
        support = {
            key: keys.count(key)
            for key in keys
            if key
        }
        variant_bonus = {
            "tight_raw": 0.30,
            "tight_processed": 0.25,
            "tight_binary": 0.20,
            "value_processed": 0.20,
            "value_detail": 0.18,
            "value_raw": 0.15,
            "value_binary": 0.10,
            "wide_processed": 0.05,
            "wide_raw": 0.0,
        }

        def score(candidate: dict[str, Any]) -> float:
            value = candidate.get("value")
            confidence = float(candidate.get("confidence") or 0.0)
            candidate_score = cls.field_candidate_score(
                field_name,
                str(value) if value is not None else None,
                confidence,
            )
            key = cls.field_candidate_key(field_name, value)
            candidate_score += support.get(key, 0) * 1.25
            candidate_score += variant_bonus.get(
                str(candidate.get("variant")),
                0.0,
            )
            if field_name in {
                "fullName",
                "placeOfOrigin",
                "placeOfResidence",
            }:
                candidate_score += cls.diacritic_score(str(value)) * 0.12
            return candidate_score

        selected = max(available, key=score)
        if field_name == "fullName":
            return cls.merge_name_candidate_accents(
                selected=selected,
                candidates=available,
            )
        return selected

    @classmethod
    def merge_name_candidate_accents(
        cls,
        selected: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Ghép dấu theo từng từ khi các lần OCR đọc cùng một họ tên."""
        value = str(selected.get("value") or "").strip()
        selected_words = value.split()
        if not selected_words:
            return selected

        equivalent = [
            candidate
            for candidate in candidates
            if cls.field_candidate_key("fullName", candidate.get("value"))
            == cls.field_candidate_key("fullName", value)
            and len(str(candidate.get("value") or "").split())
            == len(selected_words)
        ]
        if len(equivalent) < 2:
            return selected

        merged_words = list(selected_words)
        changed = False
        for index, selected_word in enumerate(selected_words):
            variants = [
                str(candidate.get("value")).split()[index]
                for candidate in equivalent
            ]
            best_score = max(cls.diacritic_score(word) for word in variants)
            richest = {
                word
                for word in variants
                if cls.diacritic_score(word) == best_score
            }
            # Khi hai biến thể có số dấu bằng nhau nhưng viết khác nhau,
            # giữ ứng viên đã chọn thay vì tự quyết định tên riêng.
            if len(richest) != 1:
                continue
            best_word = next(iter(richest))
            if (
                cls.field_candidate_key("fullName", best_word)
                == cls.field_candidate_key("fullName", selected_word)
                and cls.diacritic_score(best_word)
                > cls.diacritic_score(selected_word)
            ):
                merged_words[index] = best_word
                changed = True

        if not changed:
            return selected

        merged = dict(selected)
        merged["value"] = " ".join(merged_words)
        merged["variant"] = f"{selected.get('variant')}+accent_consensus"
        return merged

    @staticmethod
    def field_candidate_key(
        field_name: str,
        value: Any,
    ) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if field_name in {"idNumber", "dateOfBirth", "dateOfExpiry"}:
            return text
        plain = FieldOCRService.to_plain_text(text).casefold()
        plain = re.sub(r"[^a-z0-9]+", " ", plain)
        return re.sub(r"\s+", " ", plain).strip()

    @staticmethod
    def diacritic_score(value: str) -> int:
        decomposed = unicodedata.normalize("NFD", str(value))
        return sum(
            unicodedata.category(character) == "Mn"
            for character in decomposed
        ) + sum(character in "Đđ" for character in str(value))

    @staticmethod
    def field_candidate_score(
        field_name: str,
        value: str | None,
        confidence: float,
    ) -> float:
        """Chọn biến thể field đầy đủ hơn, không chỉ chọn confidence cao."""
        if not value:
            return -100.0

        text = str(value).strip()
        letters = re.findall(r"[^\W\d_]", text, flags=re.UNICODE)
        score = float(confidence) + min(len(letters), 120) / 100.0

        if field_name == "idNumber":
            return score + (100.0 if re.fullmatch(r"\d{12}", text) else 0.0)

        if field_name == "nationality":
            return score + (20.0 if text == "Việt Nam" else -5.0)

        if field_name in {"dateOfBirth", "dateOfExpiry"}:
            try:
                parsed = datetime.strptime(text, "%d/%m/%Y")
            except ValueError:
                return score - 10.0
            if not 1900 <= parsed.year <= 2100:
                return score - 10.0
            return score + 20.0

        if field_name == "fullName":
            words = text.split()
            if 2 <= len(words) <= 7:
                score += 5.0
            if any(
                label in text.casefold()
                for label in ("ngày sinh", "date of birth", "nationality")
            ):
                score -= 8.0
            return score

        if field_name in {"placeOfOrigin", "placeOfResidence"}:
            score += min(text.count(","), 4) * 0.35
            if re.search(
                r"place\s+(?:of|0f)|date\s+of|co\s+gia\s+tri",
                text,
                flags=re.IGNORECASE,
            ):
                score -= 4.0
            if re.search(r"\d{5,}|\d{5,}/\d{4}", text):
                score -= 8.0

        return score

    def prepare_portrait_result(
        self,
        portrait_result: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """
        Chuẩn hóa dữ liệu ảnh chân dung.

        Ảnh portrait không được OCR. Ảnh này được dùng cho module
        Face Verification ở bước tiếp theo.
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
        Ghép các dòng OCR của một field.

        Địa chỉ được ghép bằng dấu phẩy.
        Các trường còn lại được ghép bằng khoảng trắng.
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
        Làm sạch giá trị OCR theo loại trường.
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
        Xóa các nhãn tiếng Việt và tiếng Anh còn sót lại trong vùng OCR.
        """

        patterns = self.FIELD_LABEL_PATTERNS.get(field_name, ())
        cleaned = str(value)
        plain = self.to_plain_text(cleaned)

        target_matches = [
            match
            for pattern in patterns
            for match in re.finditer(
                pattern,
                plain,
                flags=re.IGNORECASE,
            )
        ]

        # Vùng crop cố định có thể chạm sang field kế bên. Khi nhận được
        # nhãn của field cần đọc, chỉ giữ phần sau nhãn và cắt trước nhãn
        # kế tiếp thay vì xóa nhãn rồi giữ cả dữ liệu của field hàng trên.
        if target_matches:
            target_match = min(target_matches, key=lambda item: item.start())
            cleaned = cleaned[target_match.end():]
            plain = self.to_plain_text(cleaned)

            stop_patterns: tuple[str, ...] = ()
            if field_name == "fullName":
                stop_patterns = (
                    r"\bngay\s*sinh\b",
                    r"\bdate\s*[o0][fl0]\s*birth\b",
                )
            elif field_name == "placeOfOrigin":
                stop_patterns = self.FIELD_LABEL_PATTERNS[
                    "placeOfResidence"
                ] + self.FIELD_LABEL_PATTERNS["dateOfExpiry"]
            elif field_name == "placeOfResidence":
                stop_patterns = self.FIELD_LABEL_PATTERNS["dateOfExpiry"]

            stop_matches = [
                match
                for pattern in stop_patterns
                for match in re.finditer(
                    pattern,
                    plain,
                    flags=re.IGNORECASE,
                )
            ]
            if stop_matches:
                cleaned = cleaned[
                    :min(stop_matches, key=lambda item: item.start()).start()
                ]

        for pattern in patterns:
            plain = self.to_plain_text(cleaned)
            matches = list(
                re.finditer(pattern, plain, flags=re.IGNORECASE)
            )
            for match in reversed(matches):
                cleaned = cleaned[:match.start()] + " " + cleaned[match.end():]

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
    def to_plain_text(value: str) -> str:
        normalized = unicodedata.normalize("NFD", str(value))
        plain = "".join(
            character
            for character in normalized
            if unicodedata.category(character) != "Mn"
        )
        return plain.replace("đ", "d").replace("Đ", "D")

    @staticmethod
    def clean_id_number(
        value: str,
    ) -> str | None:
        """
        Làm sạch số CCCD.

        Chỉ giữ chữ số và yêu cầu đúng 12 số.
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

        # Giữ ranh giới dòng trước khi gom toàn bộ chữ số. Vùng crop rộng
        # có thể chứa thêm một số rác ở cạnh nhưng số CCCD vẫn là một cụm
        # 12 chữ số độc lập.
        direct_match = re.search(
            r"(?<!\d)(\d{12})(?!\d)",
            normalized,
        )
        if direct_match:
            return direct_match.group(1)

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
        Làm sạch họ tên.

        Giữ chữ cái tiếng Việt và khoảng trắng.
        """

        normalized = unicodedata.normalize("NFC", str(value))
        next_label = re.search(
            r"\bn[^\W\d_]{2,5}\s+sinh\b|"
            r"\bdate\s*o[fl0]\s*birth\b",
            normalized,
            flags=re.IGNORECASE,
        )
        if next_label:
            normalized = normalized[:next_label.start()]
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
        Chuẩn hóa ngày về DD/MM/YYYY và kiểm tra ngày hợp lệ.
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

        if not 1900 <= parsed_date.year <= 2100:
            return None

        return parsed_date.strftime("%d/%m/%Y")

    @staticmethod
    def clean_gender(
        value: str,
    ) -> str | None:
        """
        Chuẩn hóa giới tính về Nam hoặc Nữ.
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

        if compact in {
            "nu",
            "no",
            "female",
            "ni",
            "nw",
            "nv",
            "nii",
        }:
            return "Nữ"

        return None

    @staticmethod
    def clean_nationality(
        value: str,
    ) -> str | None:
        """
        Chuẩn hóa quốc tịch.
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
            r"\bvie\s*nai+\b",
            r"\bvi[e3]\s*n[a-z]{2,4}\b",
            r"vi[e3][tli1].{0,2}(?:nam|nan|[il1]?amn)",
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
        Làm sạch quê quán hoặc nơi thường trú.
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
        Tính confidence trung bình của kết quả OCR.
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
        Tạo kết quả mặc định khi một field không OCR được.
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
