from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from app.modules.ocr.vietnamese_charset import (
    TEMPLATE_CHARACTERS,
    normalize_nfc,
    strip_vietnamese_marks,
    visual_candidates,
)


GLYPH_ASSET_DIR = Path(__file__).resolve().parent / "glyphs"
DEFAULT_ATLAS_PATH = GLYPH_ASSET_DIR / "vietnamese_glyph_templates.npz"
DEFAULT_MANIFEST_PATH = GLYPH_ASSET_DIR / "manifest.json"

TEXT_FIELDS = {
    "fullName",
    "gender",
    "nationality",
    "placeOfOrigin",
    "placeOfResidence",
}
NUMERIC_FIELDS = {"idNumber", "dateOfBirth", "dateOfExpiry"}

# Không sửa các từ thuộc nhãn song ngữ trên CCCD. Vùng crop rộng đôi khi có
# cả nhãn và giá trị; khóa các token này ngăn matcher thêm dấu vào tiếng Anh.
PROTECTED_LABEL_WORDS = {
    "birth",
    "date",
    "expiry",
    "full",
    "name",
    "nationality",
    "no",
    "number",
    "of",
    "origin",
    "place",
    "residence",
    "sex",
}


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    output = np.zeros_like(mask, dtype=bool)
    diameter = radius * 2 + 1
    for y in range(diameter):
        for x in range(diameter):
            output |= padded[
                y:y + mask.shape[0],
                x:x + mask.shape[1],
            ]
    return output


@dataclass(frozen=True)
class TemplateEntry:
    character: str
    font_name: str
    mask: np.ndarray
    aspect_ratio: float
    dilated_mask: np.ndarray
    pixel_count: float


@dataclass(frozen=True)
class CharacterProposal:
    character_index: int
    source: str
    target: str
    best_score: float
    original_score: float
    margin: float
    segmentation_confidence: float

    @property
    def strength(self) -> float:
        return (
            self.best_score
            + self.margin
            + max(0.0, self.best_score - self.original_score)
            + self.segmentation_confidence * 0.10
        )


class VietnameseGlyphLibrary:
    """Nạp atlas mẫu ảnh chữ/số tiếng Việt theo cách an toàn, có checksum."""

    def __init__(
        self,
        atlas_path: str | Path = DEFAULT_ATLAS_PATH,
        manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
        verify_integrity: bool = True,
    ) -> None:
        self.atlas_path = Path(atlas_path)
        self.manifest_path = Path(manifest_path)
        self.verify_integrity = bool(verify_integrity)
        self._entries: dict[str, tuple[TemplateEntry, ...]] | None = None
        self._manifest: dict[str, Any] | None = None
        self._error: str | None = None

    @property
    def available(self) -> bool:
        self._ensure_loaded()
        return self._entries is not None

    @property
    def error(self) -> str | None:
        self._ensure_loaded()
        return self._error

    @property
    def manifest(self) -> dict[str, Any]:
        self._ensure_loaded()
        return dict(self._manifest or {})

    @property
    def character_count(self) -> int:
        self._ensure_loaded()
        return len(self._entries or {})

    def has_character(self, character: str) -> bool:
        self._ensure_loaded()
        return normalize_nfc(character) in (self._entries or {})

    def entries_for(self, character: str) -> tuple[TemplateEntry, ...]:
        self._ensure_loaded()
        return (self._entries or {}).get(normalize_nfc(character), ())

    def _ensure_loaded(self) -> None:
        if self._entries is not None or self._error is not None:
            return

        try:
            manifest = json.loads(
                self.manifest_path.read_text(encoding="utf-8")
            )
            if int(manifest.get("schemaVersion", 0)) != 1:
                raise ValueError("schema atlas không được hỗ trợ")

            if self.verify_integrity:
                expected_hash = str(manifest.get("assetSha256") or "")
                actual_hash = hashlib.sha256(
                    self.atlas_path.read_bytes()
                ).hexdigest()
                if not expected_hash or actual_hash != expected_hash:
                    raise ValueError("checksum atlas không khớp")

            with np.load(self.atlas_path, allow_pickle=False) as atlas:
                templates = np.asarray(atlas["templates"], dtype=np.uint8)
                characters = np.asarray(atlas["characters"]).astype(str)
                font_names = np.asarray(atlas["font_names"]).astype(str)
                aspects = np.asarray(
                    atlas["aspect_ratios"], dtype=np.float32
                )

            item_count = int(templates.shape[0])
            if templates.ndim != 3:
                raise ValueError("templates phải có dạng N x H x W")
            if not (
                len(characters)
                == len(font_names)
                == len(aspects)
                == item_count
            ):
                raise ValueError("các mảng trong atlas không đồng bộ")

            grouped: dict[str, list[TemplateEntry]] = {}
            for index in range(item_count):
                character = normalize_nfc(characters[index])
                if len(character) != 1:
                    continue
                mask = templates[index].astype(bool)
                grouped.setdefault(character, []).append(
                    TemplateEntry(
                        character=character,
                        font_name=font_names[index],
                        mask=mask,
                        aspect_ratio=float(aspects[index]),
                        dilated_mask=_dilate_mask(mask, radius=1),
                        pixel_count=float(mask.sum()),
                    )
                )

            expected = set(TEMPLATE_CHARACTERS)
            missing = sorted(expected - set(grouped))
            if missing:
                raise ValueError(
                    "atlas thiếu ký tự: " + " ".join(missing[:20])
                )

            self._entries = {
                character: tuple(entries)
                for character, entries in grouped.items()
            }
            self._manifest = manifest
        except Exception as error:
            self._entries = None
            self._manifest = None
            self._error = f"Không thể nạp thư viện glyph: {error}"


class VietnameseGlyphMatcher:
    """
    Đối chiếu ký tự trong OCR box với atlas chữ/số tiếng Việt.

    Matcher chỉ sửa khi ảnh đủ lớn, tách được đúng số ký tự và phương án tốt
    nhất vượt ngưỡng tuyệt đối, ngưỡng chênh lệch và điểm ký tự OCR ban đầu.
    Khi bằng chứng yếu, văn bản OCR được giữ nguyên.
    """

    def __init__(
        self,
        library: VietnameseGlyphLibrary | None = None,
        minimum_character_height: int = 16,
        text_minimum_score: float = 0.78,
        text_minimum_margin: float = 0.020,
        text_minimum_improvement: float = 0.045,
        numeric_minimum_score: float = 0.76,
        numeric_minimum_margin: float = 0.075,
        numeric_minimum_improvement: float = 0.070,
        enabled: bool = True,
        auto_correct: bool = True,
    ) -> None:
        self.library = library or VietnameseGlyphLibrary()
        self.minimum_character_height = int(minimum_character_height)
        self.text_minimum_score = float(text_minimum_score)
        self.text_minimum_margin = float(text_minimum_margin)
        self.text_minimum_improvement = float(
            text_minimum_improvement
        )
        self.numeric_minimum_score = float(numeric_minimum_score)
        self.numeric_minimum_margin = float(numeric_minimum_margin)
        self.numeric_minimum_improvement = float(
            numeric_minimum_improvement
        )
        self.enabled = bool(enabled)
        self.auto_correct = bool(auto_correct)

    def refine_ocr_boxes(
        self,
        image_path: str | Path,
        text_boxes: list[Any],
        field_name: str,
    ) -> tuple[dict[int, str], dict[str, Any]]:
        """Trả về text override theo chỉ số box và báo cáo JSON-safe."""
        report: dict[str, Any] = {
            "enabled": self.enabled,
            "autoCorrect": self.auto_correct,
            "available": False,
            "fieldName": field_name,
            "atlasVersion": None,
            "boxCount": len(text_boxes),
            "processedBoxes": 0,
            "comparedCharacters": 0,
            "confidentCharacters": 0,
            "averageBestScore": 0.0,
            "coverage": 0.0,
            "applied": False,
            "corrections": [],
            "reviewCandidates": [],
            "skippedReason": None,
        }

        if not self.enabled:
            report["skippedReason"] = "DISABLED"
            return {}, report

        if field_name not in TEXT_FIELDS | NUMERIC_FIELDS:
            report["skippedReason"] = "FIELD_NOT_SUPPORTED"
            return {}, report

        if not self.library.available:
            report["skippedReason"] = "ATLAS_UNAVAILABLE"
            report["error"] = self.library.error
            return {}, report

        report["available"] = True
        report["atlasVersion"] = self.library.manifest.get(
            "libraryVersion"
        )

        try:
            with Image.open(image_path) as opened:
                image = ImageOps.exif_transpose(opened).convert("L")
        except Exception as error:
            report["skippedReason"] = "IMAGE_DECODE_FAILED"
            report["error"] = str(error)
            return {}, report

        overrides: dict[int, str] = {}
        scores: list[float] = []
        possible = 0

        for box_index, text_box in enumerate(text_boxes):
            text = normalize_nfc(str(getattr(text_box, "text", "")).strip())
            box = getattr(text_box, "box", None)
            if not text or not box:
                continue

            refined = self._refine_box(
                image=image,
                box=box,
                text=text,
                field_name=field_name,
            )
            if refined is None:
                continue

            report["processedBoxes"] += 1
            report["comparedCharacters"] += refined["compared"]
            report["confidentCharacters"] += refined["confident"]
            possible += refined["possible"]
            scores.extend(refined["scores"])

            corrected_text = str(refined["text"])
            if corrected_text != text:
                overrides[box_index] = corrected_text

            for correction in refined["corrections"]:
                report["corrections"].append(
                    {
                        "boxIndex": box_index,
                        **correction,
                    }
                )
            for review in refined["reviewCandidates"]:
                if len(report["reviewCandidates"]) >= 24:
                    break
                report["reviewCandidates"].append(
                    {
                        "boxIndex": box_index,
                        **review,
                    }
                )

        compared = int(report["comparedCharacters"])
        report["averageBestScore"] = round(
            float(sum(scores) / len(scores)) if scores else 0.0,
            6,
        )
        report["coverage"] = round(
            compared / possible if possible else 0.0,
            6,
        )
        report["applied"] = bool(overrides)

        if report["processedBoxes"] == 0:
            report["skippedReason"] = "NO_USABLE_TEXT_BOX"
        elif compared == 0:
            report["skippedReason"] = "NO_RELIABLE_SEGMENTATION"
        elif not overrides:
            report["skippedReason"] = (
                "REPORT_ONLY"
                if not self.auto_correct
                else "NO_HIGH_CONFIDENCE_CHANGE"
            )

        return overrides, report

    def _refine_box(
        self,
        image: Image.Image,
        box: Any,
        text: str,
        field_name: str,
    ) -> dict[str, Any] | None:
        crop = self._crop_box(image=image, box=box)
        if crop is None or crop.height < self.minimum_character_height:
            return None

        mask = self._to_binary_mask(crop)
        character_indices = [
            index
            for index, character in enumerate(text)
            if not character.isspace()
        ]
        if not character_indices:
            return None

        segments = self._segment_characters(
            mask=mask,
            expected_count=len(character_indices),
        )
        if segments is None:
            return None

        segment_masks, segmentation_confidence = segments
        protected = self._protected_character_indices(text)
        proposals: list[CharacterProposal] = []
        scores: list[float] = []
        compared = 0
        confident = 0
        possible = 0
        review_candidates: list[dict[str, Any]] = []

        for character_index, segment in zip(
            character_indices,
            segment_masks,
            strict=True,
        ):
            source = text[character_index]
            candidates = tuple(
                candidate
                for candidate in visual_candidates(source, field_name)
                if self.library.has_character(candidate)
            )
            if len(candidates) <= 1:
                continue
            possible += 1
            if character_index in protected:
                continue

            ranked = self.rank_candidates(segment, candidates)
            if not ranked:
                continue
            compared += 1
            best_character, best_score = ranked[0]
            scores.append(best_score)

            score_by_character = dict(ranked)
            original_score = float(score_by_character.get(source, 0.0))
            second_score = ranked[1][1] if len(ranked) > 1 else 0.0
            margin = best_score - second_score

            minimum_score, minimum_margin, minimum_improvement = (
                self._thresholds(field_name)
            )
            if best_score >= minimum_score:
                confident += 1

            if best_character == source:
                continue
            rejection_reason: str | None = None
            if segmentation_confidence < 0.70:
                rejection_reason = "LOW_SEGMENTATION_CONFIDENCE"
            elif best_score < minimum_score:
                rejection_reason = "LOW_TEMPLATE_SCORE"
            elif margin < minimum_margin:
                rejection_reason = "AMBIGUOUS_TOP_CANDIDATES"

            required_improvement = minimum_improvement
            missing_mark_upgrade = self._is_missing_mark_upgrade(
                source,
                best_character,
            )
            if missing_mark_upgrade:
                # Khi EasyOCR trả chữ không dấu, phần thân ký tự vẫn khớp rất
                # cao với cả hai mẫu. Cho phép mức cải thiện nhỏ hơn nhưng vẫn
                # giữ ngưỡng score và margin để dấu nhiễu không được thêm vào.
                if best_score < 0.88:
                    rejection_reason = "WEAK_MISSING_MARK_EVIDENCE"
                required_improvement = min(required_improvement, 0.015)
            if (
                rejection_reason is None
                and best_score - original_score < required_improvement
            ):
                rejection_reason = "LOW_IMPROVEMENT_OVER_OCR"

            review_candidates.append(
                {
                    "characterIndex": character_index,
                    "from": source,
                    "bestCandidate": best_character,
                    "bestScore": round(best_score, 6),
                    "originalScore": round(original_score, 6),
                    "margin": round(margin, 6),
                    "decision": rejection_reason or "PROPOSED",
                    "top3": [
                        {
                            "character": character,
                            "score": round(score, 6),
                        }
                        for character, score in ranked[:3]
                    ],
                }
            )

            if rejection_reason is None:
                proposals.append(
                    CharacterProposal(
                        character_index=character_index,
                        source=source,
                        target=best_character,
                        best_score=best_score,
                        original_score=original_score,
                        margin=margin,
                        segmentation_confidence=segmentation_confidence,
                    )
                )

        # Một box bị phân đoạn sai thường đề xuất đổi hàng loạt. Chỉ áp dụng
        # các đề xuất mạnh nhất. Box tách rất rõ được đổi tối đa 75%, còn box
        # yếu chỉ 45% số ký tự đã so sánh.
        change_ratio = 0.75 if segmentation_confidence >= 0.85 else 0.45
        maximum_changes = max(
            2,
            int(math.ceil(max(compared, 1) * change_ratio)),
        )
        accepted = sorted(
            proposals,
            key=lambda item: item.strength,
            reverse=True,
        )[:maximum_changes]
        if not self.auto_correct:
            accepted = []
        accepted.sort(key=lambda item: item.character_index)
        accepted_indices = {
            item.character_index
            for item in accepted
        }
        for review in review_candidates:
            if review["decision"] != "PROPOSED":
                continue
            review["decision"] = (
                "APPLIED"
                if review["characterIndex"] in accepted_indices
                else (
                    "REPORT_ONLY"
                    if not self.auto_correct
                    else "CHANGE_LIMIT"
                )
            )

        output = list(text)
        corrections: list[dict[str, Any]] = []
        for proposal in accepted:
            output[proposal.character_index] = proposal.target
            corrections.append(
                {
                    "characterIndex": proposal.character_index,
                    "from": proposal.source,
                    "to": proposal.target,
                    "score": round(proposal.best_score, 6),
                    "originalScore": round(proposal.original_score, 6),
                    "margin": round(proposal.margin, 6),
                    "segmentationConfidence": round(
                        proposal.segmentation_confidence,
                        6,
                    ),
                }
            )

        return {
            "text": "".join(output),
            "possible": possible,
            "compared": compared,
            "confident": confident,
            "scores": scores,
            "corrections": corrections,
            "reviewCandidates": review_candidates[:12],
        }

    def rank_candidates(
        self,
        glyph_mask: np.ndarray,
        candidates: tuple[str, ...] | list[str],
    ) -> list[tuple[str, float]]:
        """Chấm điểm từng ký tự, lấy mẫu font tốt nhất cho mỗi ký tự."""
        query, query_aspect = self._normalize_glyph(glyph_mask)
        if query is None:
            return []

        ranked: list[tuple[str, float]] = []
        query_count = float(query.sum())
        dilated_query = _dilate_mask(query, radius=1)
        for character in dict.fromkeys(candidates):
            entries = self.library.entries_for(character)
            if not entries:
                continue
            score = max(
                self._template_similarity(
                    query=query,
                    query_aspect=query_aspect,
                    query_count=query_count,
                    dilated_query=dilated_query,
                    entry=entry,
                )
                for entry in entries
            )
            ranked.append((character, float(score)))

        return sorted(ranked, key=lambda item: item[1], reverse=True)

    def _thresholds(self, field_name: str) -> tuple[float, float, float]:
        if field_name in NUMERIC_FIELDS:
            return (
                self.numeric_minimum_score,
                self.numeric_minimum_margin,
                self.numeric_minimum_improvement,
            )
        return (
            self.text_minimum_score,
            self.text_minimum_margin,
            self.text_minimum_improvement,
        )

    @staticmethod
    def _is_missing_mark_upgrade(source: str, target: str) -> bool:
        source_plain = strip_vietnamese_marks(source)
        target_plain = strip_vietnamese_marks(target)
        return bool(
            source_plain == target_plain
            and source == source_plain
            and target != target_plain
        )

    @staticmethod
    def _crop_box(image: Image.Image, box: Any) -> Image.Image | None:
        try:
            points = np.asarray(box, dtype=np.float32).reshape(-1, 2)
        except (TypeError, ValueError):
            return None
        if len(points) < 2 or not np.isfinite(points).all():
            return None

        x1 = int(math.floor(float(points[:, 0].min())))
        y1 = int(math.floor(float(points[:, 1].min())))
        x2 = int(math.ceil(float(points[:, 0].max())))
        y2 = int(math.ceil(float(points[:, 1].max())))
        height = max(0, y2 - y1)
        padding = max(1, int(round(height * 0.04)))
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.width, x2 + padding)
        y2 = min(image.height, y2 + padding)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        return image.crop((x1, y1, x2, y2))

    @staticmethod
    def _otsu_threshold(gray: np.ndarray) -> int:
        histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
        total = float(gray.size)
        weighted_total = float(np.dot(np.arange(256), histogram))
        background_weight = 0.0
        background_sum = 0.0
        best_variance = -1.0
        best_threshold = 127

        for threshold in range(256):
            background_weight += histogram[threshold]
            if background_weight <= 0:
                continue
            foreground_weight = total - background_weight
            if foreground_weight <= 0:
                break
            background_sum += threshold * histogram[threshold]
            background_mean = background_sum / background_weight
            foreground_mean = (
                weighted_total - background_sum
            ) / foreground_weight
            variance = (
                background_weight
                * foreground_weight
                * (background_mean - foreground_mean) ** 2
            )
            if variance > best_variance:
                best_variance = variance
                best_threshold = threshold
        return int(best_threshold)

    @classmethod
    def _to_binary_mask(cls, image: Image.Image) -> np.ndarray:
        gray = np.asarray(ImageOps.autocontrast(image), dtype=np.uint8)
        threshold = cls._otsu_threshold(gray)
        border = np.concatenate(
            (gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1])
        )
        dark_text = float(np.median(border)) >= float(np.median(gray))
        mask = gray <= threshold if dark_text else gray > threshold

        ratio = float(mask.mean())
        if ratio > 0.55:
            mask = ~mask
        if float(mask.mean()) < 0.003:
            fallback = int(np.percentile(gray, 30))
            mask = gray <= fallback

        # Loại điểm nhiễu đơn lẻ nhưng giữ dấu thanh có ít nhất hai pixel.
        padded = np.pad(mask, 1, mode="constant", constant_values=False)
        neighbours = np.zeros(mask.shape, dtype=np.uint8)
        for row_offset in range(3):
            for column_offset in range(3):
                neighbours += padded[
                    row_offset:row_offset + mask.shape[0],
                    column_offset:column_offset + mask.shape[1],
                ]
        return mask & (neighbours >= 2)

    @classmethod
    def _segment_characters(
        cls,
        mask: np.ndarray,
        expected_count: int,
    ) -> tuple[list[np.ndarray], float] | None:
        if mask.ndim != 2 or expected_count <= 0:
            return None
        row_projection = mask.sum(axis=1)
        active_rows = np.flatnonzero(row_projection > 0)
        if active_rows.size == 0:
            return None
        y1 = max(0, int(active_rows[0]) - 1)
        y2 = min(mask.shape[0], int(active_rows[-1]) + 2)
        line = mask[y1:y2, :]

        column_projection = line.sum(axis=0)
        minimum_pixels = max(1, int(round(line.shape[0] * 0.018)))
        active = column_projection >= minimum_pixels
        runs = cls._boolean_runs(active)
        if not runs:
            return None

        initial_count = len(runs)
        if abs(initial_count - expected_count) > max(5, expected_count // 2):
            return None

        adjusted = [list(run) for run in runs]
        while len(adjusted) > expected_count:
            merge_index = min(
                range(len(adjusted) - 1),
                key=lambda index: (
                    adjusted[index + 1][0] - adjusted[index][1],
                    min(
                        adjusted[index][1] - adjusted[index][0],
                        adjusted[index + 1][1] - adjusted[index + 1][0],
                    ),
                ),
            )
            adjusted[merge_index:merge_index + 2] = [[
                adjusted[merge_index][0],
                adjusted[merge_index + 1][1],
            ]]

        while len(adjusted) < expected_count:
            split = cls._best_split(adjusted, column_projection)
            if split is None:
                return None
            segment_index, cut = split
            start, end = adjusted[segment_index]
            adjusted[segment_index:segment_index + 1] = [
                [start, cut],
                [cut, end],
            ]

        segment_masks: list[np.ndarray] = []
        for start, end in adjusted:
            x1 = max(0, int(start) - 1)
            x2 = min(line.shape[1], int(end) + 1)
            segment = line[:, x1:x2]
            if segment.size == 0 or not segment.any():
                return None
            segment_masks.append(segment)

        widths = np.asarray(
            [segment.shape[1] for segment in segment_masks],
            dtype=np.float32,
        )
        width_variation = float(widths.std() / max(widths.mean(), 1.0))
        count_penalty = abs(initial_count - expected_count) / expected_count
        confidence = 1.0 - min(0.45, count_penalty * 0.50)
        confidence -= min(0.25, width_variation * 0.16)
        confidence = float(max(0.0, min(1.0, confidence)))
        return segment_masks, confidence

    @staticmethod
    def _boolean_runs(active: np.ndarray) -> list[tuple[int, int]]:
        runs: list[tuple[int, int]] = []
        start: int | None = None
        for index, value in enumerate(active.tolist() + [False]):
            if value and start is None:
                start = index
            elif not value and start is not None:
                runs.append((start, index))
                start = None
        return runs

    @staticmethod
    def _best_split(
        segments: list[list[int]],
        projection: np.ndarray,
    ) -> tuple[int, int] | None:
        choices: list[tuple[float, int, int]] = []
        for index, (start, end) in enumerate(segments):
            width = end - start
            if width < 6:
                continue
            left = start + max(2, int(round(width * 0.25)))
            right = end - max(2, int(round(width * 0.25)))
            if right <= left:
                continue
            local = projection[left:right]
            cut = left + int(np.argmin(local))
            valley = float(projection[cut])
            score = width / (1.0 + valley)
            choices.append((score, index, cut))
        if not choices:
            return None
        _, index, cut = max(choices)
        return index, cut

    @staticmethod
    def _normalize_glyph(
        mask: np.ndarray,
        canvas_size: int = 64,
        margin: int = 4,
    ) -> tuple[np.ndarray | None, float]:
        rows, columns = np.where(mask)
        if rows.size == 0 or columns.size == 0:
            return None, 0.0
        cropped = mask[
            int(rows.min()):int(rows.max()) + 1,
            int(columns.min()):int(columns.max()) + 1,
        ]
        height, width = cropped.shape
        if height < 2 or width < 1:
            return None, 0.0
        aspect = float(width / height)

        available = canvas_size - margin * 2
        scale = min(available / width, available / height)
        target_width = max(1, int(round(width * scale)))
        target_height = max(1, int(round(height * scale)))
        resized = Image.fromarray(cropped.astype(np.uint8) * 255).resize(
            (target_width, target_height),
            Image.Resampling.NEAREST,
        )
        normalized = np.zeros((canvas_size, canvas_size), dtype=bool)
        x = (canvas_size - target_width) // 2
        y = (canvas_size - target_height) // 2
        normalized[
            y:y + target_height,
            x:x + target_width,
        ] = np.asarray(resized, dtype=np.uint8) > 0
        return normalized, aspect

    @staticmethod
    def _template_similarity(
        query: np.ndarray,
        query_aspect: float,
        query_count: float,
        dilated_query: np.ndarray,
        entry: TemplateEntry,
    ) -> float:
        template = entry.mask
        template_count = entry.pixel_count
        if query_count <= 0 or template_count <= 0:
            return 0.0

        query_hit = float((query & entry.dilated_mask).sum()) / query_count
        template_hit = (
            float((template & dilated_query).sum()) / template_count
        )
        tolerant = (
            2.0 * query_hit * template_hit / (query_hit + template_hit)
            if query_hit + template_hit > 0
            else 0.0
        )

        intersection = float((query & template).sum())
        cosine = intersection / math.sqrt(query_count * template_count)
        aspect_distance = abs(
            math.log(
                max(query_aspect, 1e-6)
                / max(entry.aspect_ratio, 1e-6)
            )
        )
        aspect_score = math.exp(-0.65 * aspect_distance)
        density_score = math.exp(
            -1.5
            * abs(
                math.log(
                    max(query_count, 1.0) / max(template_count, 1.0)
                )
            )
        )
        score = (
            tolerant * 0.66
            + cosine * 0.18
            + aspect_score * 0.10
            + density_score * 0.06
        )
        return float(max(0.0, min(1.0, score)))

    @staticmethod
    def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
        return _dilate_mask(mask, radius)

    @staticmethod
    def _protected_character_indices(text: str) -> set[int]:
        protected: set[int] = set()
        for match in re.finditer(r"[^\W\d_]+", text, flags=re.UNICODE):
            token = match.group(0).casefold()
            if token in PROTECTED_LABEL_WORDS:
                protected.update(range(match.start(), match.end()))
        return protected


def _environment_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() not in {"0", "false", "no", "off"}


vietnamese_glyph_matcher = VietnameseGlyphMatcher(
    enabled=_environment_flag("OCR_GLYPH_MATCHER_ENABLED", True),
    auto_correct=_environment_flag("OCR_GLYPH_AUTO_CORRECT", True),
)


__all__ = [
    "DEFAULT_ATLAS_PATH",
    "DEFAULT_MANIFEST_PATH",
    "VietnameseGlyphLibrary",
    "VietnameseGlyphMatcher",
    "vietnamese_glyph_matcher",
]
