from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, Sequence

from app.modules.ocr.layout.geometry import (
    is_below,
    is_to_right_of,
    score_below_candidate,
    score_right_candidate,
)
from app.modules.ocr.layout.text_box import (
    LayoutTextBox,
)


@dataclass(frozen=True, slots=True)
class AnchorMatch:
    """
    Kết quả so khớp một OCR box với nhãn cần tìm.
    """

    anchor_name: str
    matched_alias: str
    box: LayoutTextBox
    similarity: float

    def to_dict(self) -> dict:
        return {
            "anchorName": self.anchor_name,
            "matchedAlias": self.matched_alias,
            "similarity": round(
                self.similarity,
                4,
            ),
            "box": self.box.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CandidateMatch:
    """
    Kết quả lựa chọn giá trị dựa trên một anchor.
    """

    anchor: AnchorMatch
    candidate: LayoutTextBox
    direction: str
    geometry_score: float
    final_score: float

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "geometryScore": round(
                self.geometry_score,
                4,
            ),
            "finalScore": round(
                self.final_score,
                4,
            ),
            "anchor": self.anchor.to_dict(),
            "candidate": self.candidate.to_dict(),
        }


def remove_accents(text: str) -> str:
    """
    Bỏ dấu tiếng Việt.

    Ví dụ:
        Họ và tên -> Ho va ten
        Quốc tịch -> Quoc tich
    """

    normalized = unicodedata.normalize(
        "NFD",
        text,
    )

    without_accents = "".join(
        character
        for character in normalized
        if unicodedata.category(
            character
        )
        != "Mn"
    )

    return (
        without_accents
        .replace("đ", "d")
        .replace("Đ", "D")
    )


def normalize_anchor_text(
    text: str,
) -> str:
    """
    Chuẩn hóa văn bản để so khớp anchor.

    Quy trình:
    - Bỏ dấu.
    - Chuyển chữ thường.
    - Thay ký tự đặc biệt bằng khoảng trắng.
    - Xóa khoảng trắng thừa.
    """

    normalized = remove_accents(
        text
    ).lower()

    normalized = re.sub(
        r"[^a-z0-9]+",
        " ",
        normalized,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()

    return normalized


def compact_anchor_text(
    text: str,
) -> str:
    """
    Chuẩn hóa anchor và xóa toàn bộ khoảng trắng.

    Hữu ích khi OCR đọc dính chữ:

        Hovaten
        Ngaysinh
        Quoctich
    """

    return normalize_anchor_text(
        text
    ).replace(" ", "")


def calculate_text_similarity(
    first: str,
    second: str,
) -> float:
    """
    Tính độ giống nhau giữa hai chuỗi.

    Hàm kết hợp:
    - So khớp chuỗi có khoảng trắng.
    - So khớp chuỗi đã xóa khoảng trắng.
    - Kiểm tra một chuỗi có chứa chuỗi kia.
    """

    normalized_first = (
        normalize_anchor_text(first)
    )

    normalized_second = (
        normalize_anchor_text(second)
    )

    if (
        not normalized_first
        or not normalized_second
    ):
        return 0.0

    if normalized_first == normalized_second:
        return 1.0

    compact_first = (
        normalized_first.replace(" ", "")
    )

    compact_second = (
        normalized_second.replace(" ", "")
    )

    if compact_first == compact_second:
        return 1.0

    normal_similarity = (
        SequenceMatcher(
            None,
            normalized_first,
            normalized_second,
        ).ratio()
    )

    compact_similarity = (
        SequenceMatcher(
            None,
            compact_first,
            compact_second,
        ).ratio()
    )

    containment_score = 0.0

    if (
        normalized_first
        in normalized_second
        or normalized_second
        in normalized_first
    ):
        shorter_length = min(
            len(normalized_first),
            len(normalized_second),
        )

        longer_length = max(
            len(normalized_first),
            len(normalized_second),
        )

        if longer_length > 0:
            containment_score = (
                shorter_length
                / longer_length
            )

    if (
        compact_first in compact_second
        or compact_second in compact_first
    ):
        shorter_length = min(
            len(compact_first),
            len(compact_second),
        )

        longer_length = max(
            len(compact_first),
            len(compact_second),
        )

        if longer_length > 0:
            containment_score = max(
                containment_score,
                shorter_length
                / longer_length,
            )

    return max(
        normal_similarity,
        compact_similarity,
        containment_score,
    )


class AnchorFinder:
    """
    Tìm nhãn và giá trị dựa trên vị trí OCR box.
    """

    def __init__(
        self,
        boxes: Sequence[LayoutTextBox],
        minimum_anchor_similarity: float = 0.60,
    ) -> None:
        self.boxes = list(boxes)
        self.minimum_anchor_similarity = (
            minimum_anchor_similarity
        )

    def find_anchor(
        self,
        anchor_name: str,
        aliases: Iterable[str],
        minimum_similarity: float | None = None,
    ) -> AnchorMatch | None:
        """
        Tìm box phù hợp nhất với một nhóm alias.

        Ví dụ aliases:

        [
            "Họ và tên",
            "Ho va ten",
            "Full name",
        ]
        """

        threshold = (
            minimum_similarity
            if minimum_similarity is not None
            else self.minimum_anchor_similarity
        )

        best_match: AnchorMatch | None = None

        alias_list = [
            alias
            for alias in aliases
            if alias.strip()
        ]

        for box in self.boxes:
            for alias in alias_list:
                similarity = (
                    calculate_text_similarity(
                        box.text,
                        alias,
                    )
                )

                if similarity < threshold:
                    continue

                candidate_match = AnchorMatch(
                    anchor_name=anchor_name,
                    matched_alias=alias,
                    box=box,
                    similarity=similarity,
                )

                if best_match is None:
                    best_match = (
                        candidate_match
                    )
                    continue

                if (
                    candidate_match.similarity
                    > best_match.similarity
                ):
                    best_match = (
                        candidate_match
                    )
                    continue

                if (
                    candidate_match.similarity
                    == best_match.similarity
                    and candidate_match.box.confidence
                    > best_match.box.confidence
                ):
                    best_match = (
                        candidate_match
                    )

        return best_match

    def find_all_anchors(
        self,
        anchor_name: str,
        aliases: Iterable[str],
        minimum_similarity: float | None = None,
    ) -> list[AnchorMatch]:
        """
        Trả về toàn bộ anchor phù hợp, sắp xếp theo điểm.
        """

        threshold = (
            minimum_similarity
            if minimum_similarity is not None
            else self.minimum_anchor_similarity
        )

        results: list[AnchorMatch] = []

        for box in self.boxes:
            best_alias = ""
            best_similarity = 0.0

            for alias in aliases:
                similarity = (
                    calculate_text_similarity(
                        box.text,
                        alias,
                    )
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_alias = alias

            if best_similarity < threshold:
                continue

            results.append(
                AnchorMatch(
                    anchor_name=anchor_name,
                    matched_alias=best_alias,
                    box=box,
                    similarity=best_similarity,
                )
            )

        return sorted(
            results,
            key=lambda item: (
                item.similarity,
                item.box.confidence,
            ),
            reverse=True,
        )

    def find_right_candidate(
        self,
        anchor: AnchorMatch,
        excluded_boxes: Iterable[
            LayoutTextBox
        ] | None = None,
        maximum_gap_ratio: float = 8.0,
    ) -> CandidateMatch | None:
        """
        Tìm box phù hợp nhất nằm bên phải anchor.
        """

        excluded_ids = {
            id(box)
            for box in (
                excluded_boxes or []
            )
        }

        best_match: CandidateMatch | None = None

        for candidate in self.boxes:
            if candidate is anchor.box:
                continue

            if id(candidate) in excluded_ids:
                continue

            if not is_to_right_of(
                candidate,
                anchor.box,
            ):
                continue

            geometry_score = (
                score_right_candidate(
                    anchor=anchor.box,
                    candidate=candidate,
                    maximum_gap_ratio=(
                        maximum_gap_ratio
                    ),
                )
            )

            if geometry_score == float("-inf"):
                continue

            final_score = (
                geometry_score
                + anchor.similarity * 2.0
            )

            current_match = CandidateMatch(
                anchor=anchor,
                candidate=candidate,
                direction="right",
                geometry_score=geometry_score,
                final_score=final_score,
            )

            if (
                best_match is None
                or current_match.final_score
                > best_match.final_score
            ):
                best_match = current_match

        return best_match

    def find_below_candidate(
        self,
        anchor: AnchorMatch,
        excluded_boxes: Iterable[
            LayoutTextBox
        ] | None = None,
        maximum_gap_ratio: float = 5.0,
    ) -> CandidateMatch | None:
        """
        Tìm box phù hợp nhất nằm phía dưới anchor.
        """

        excluded_ids = {
            id(box)
            for box in (
                excluded_boxes or []
            )
        }

        best_match: CandidateMatch | None = None

        for candidate in self.boxes:
            if candidate is anchor.box:
                continue

            if id(candidate) in excluded_ids:
                continue

            if not is_below(
                candidate,
                anchor.box,
            ):
                continue

            geometry_score = (
                score_below_candidate(
                    anchor=anchor.box,
                    candidate=candidate,
                    maximum_gap_ratio=(
                        maximum_gap_ratio
                    ),
                )
            )

            if geometry_score == float("-inf"):
                continue

            final_score = (
                geometry_score
                + anchor.similarity * 2.0
            )

            current_match = CandidateMatch(
                anchor=anchor,
                candidate=candidate,
                direction="below",
                geometry_score=geometry_score,
                final_score=final_score,
            )

            if (
                best_match is None
                or current_match.final_score
                > best_match.final_score
            ):
                best_match = current_match

        return best_match

    def find_value(
        self,
        anchor_name: str,
        aliases: Iterable[str],
        preferred_direction: str = "right",
        fallback_direction: str | None = "below",
        minimum_similarity: float | None = None,
        excluded_boxes: Iterable[
            LayoutTextBox
        ] | None = None,
    ) -> CandidateMatch | None:
        """
        Tìm anchor và giá trị tương ứng.

        preferred_direction:
            right hoặc below

        fallback_direction:
            Hướng dự phòng nếu không tìm được ứng viên.
        """

        anchor = self.find_anchor(
            anchor_name=anchor_name,
            aliases=aliases,
            minimum_similarity=(
                minimum_similarity
            ),
        )

        if anchor is None:
            return None

        preferred_result = (
            self._find_by_direction(
                anchor=anchor,
                direction=preferred_direction,
                excluded_boxes=excluded_boxes,
            )
        )

        if preferred_result is not None:
            return preferred_result

        if (
            fallback_direction is None
            or fallback_direction
            == preferred_direction
        ):
            return None

        return self._find_by_direction(
            anchor=anchor,
            direction=fallback_direction,
            excluded_boxes=excluded_boxes,
        )

    def _find_by_direction(
        self,
        anchor: AnchorMatch,
        direction: str,
        excluded_boxes: Iterable[
            LayoutTextBox
        ] | None = None,
    ) -> CandidateMatch | None:
        """
        Gọi phương thức tìm candidate theo hướng.
        """

        normalized_direction = (
            direction.strip().lower()
        )

        if normalized_direction == "right":
            return self.find_right_candidate(
                anchor=anchor,
                excluded_boxes=excluded_boxes,
            )

        if normalized_direction == "below":
            return self.find_below_candidate(
                anchor=anchor,
                excluded_boxes=excluded_boxes,
            )

        raise ValueError(
            "direction chỉ hỗ trợ "
            "'right' hoặc 'below'"
        )
