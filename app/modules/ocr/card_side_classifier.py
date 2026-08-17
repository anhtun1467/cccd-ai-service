from __future__ import annotations

import re
from typing import Any

from app.modules.ocr.result_fuser import remove_accents


FRONT_PATTERNS: tuple[tuple[str, float, str], ...] = (
    (r"can\s*cuoc\s*cong\s*dan", 3.0, "CITIZEN_IDENTITY_TITLE"),
    (r"citizen\s*identity\s*card", 2.0, "CITIZEN_IDENTITY_ENGLISH"),
    (r"ho\s*va\s*ten|full\s*name", 2.0, "FULL_NAME_LABEL"),
    (r"ngay\s*sinh|date\s*of\s*birth", 2.0, "DATE_OF_BIRTH_LABEL"),
    (r"gioi\s*tinh|\bsex\b", 1.0, "GENDER_LABEL"),
    (r"quoc\s*tich|nationality", 1.0, "NATIONALITY_LABEL"),
)

BACK_PATTERNS: tuple[tuple[str, float, str], ...] = (
    (r"noi\s*dang\s*ky\s*khai\s*sinh|place\s*of\s*birth", 2.5, "PLACE_OF_BIRTH_LABEL"),
    (r"ngay\s*thang\s*nam\s*cap|date\s*of\s*issue", 3.0, "DATE_OF_ISSUE_LABEL"),
    (r"bo\s*cong\s*an|ministry\s*of\s*public\s*security", 2.5, "ISSUER_LABEL"),
    (r"\bidvnm", 3.0, "MRZ_PREFIX"),
    (r"<{4,}", 2.0, "MRZ_FILLER"),
)


def classify_cccd_side(ocr_result: dict[str, Any] | None) -> dict[str, Any]:
    result = ocr_result or {}
    lines = result.get("normalizedText", result.get("rawText", []))
    if isinstance(lines, str):
        lines = lines.splitlines()
    if not isinstance(lines, list):
        lines = []

    text = remove_accents(
        " ".join(str(line) for line in lines if line)
    ).casefold()
    text = re.sub(r"\s+", " ", text)

    def score_patterns(
        patterns: tuple[tuple[str, float, str], ...],
    ) -> tuple[float, list[str]]:
        score = 0.0
        evidence: list[str] = []
        for pattern, weight, label in patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                score += weight
                evidence.append(label)
        return round(score, 2), evidence

    front_score, front_evidence = score_patterns(FRONT_PATTERNS)
    back_score, back_evidence = score_patterns(BACK_PATTERNS)

    if back_score >= 4.0 and back_score >= front_score + 1.5:
        side = "BACK"
        confidence = min(1.0, 0.55 + (back_score - front_score) * 0.06)
    elif front_score >= 4.0 and front_score >= back_score + 1.0:
        side = "FRONT"
        confidence = min(1.0, 0.55 + (front_score - back_score) * 0.05)
    else:
        side = "UNKNOWN"
        confidence = 0.0

    return {
        "side": side,
        "confidence": round(float(confidence), 3),
        "frontScore": front_score,
        "backScore": back_score,
        "frontEvidence": front_evidence,
        "backEvidence": back_evidence,
    }
