from __future__ import annotations

import numpy as np

from app.modules.face_verification.errors import FaceVerificationError
from app.modules.face_verification.models import FaceEmbeddingResult
from app.modules.face_verification.portrait_extractor import (
    CCCDPortraitExtractor,
)


class SequenceAnalyzer:
    def __init__(self, results: list[list[FaceEmbeddingResult]]) -> None:
        self.results = list(results)
        self.call_count = 0

    def extract(self, image: np.ndarray) -> list[FaceEmbeddingResult]:
        self.call_count += 1
        if not self.results:
            return []
        return self.results.pop(0)


def make_face() -> FaceEmbeddingResult:
    return FaceEmbeddingResult(
        embedding=np.ones(512, dtype=np.float32),
        bbox=np.asarray([80, 90, 230, 300], dtype=np.float32),
        detection_score=0.97,
        landmarks=None,
        pose=None,
    )


def test_portrait_roi_reuses_embedding_from_first_detection() -> None:
    analyzer = SequenceAnalyzer([[make_face()]])
    extractor = CCCDPortraitExtractor(analyzer=analyzer)
    card = np.full((630, 1000, 3), 128, dtype=np.uint8)

    result = extractor.extract(card)

    assert result.extraction_method == "portrait_roi_original"
    assert result.embedding_result is not None
    assert result.embedding_result.dimension == 512
    assert analyzer.call_count == 1
    assert result.bbox[0] > 0
    assert result.portrait.size > 0


def test_missing_card_portrait_has_stable_error_code() -> None:
    analyzer = SequenceAnalyzer([])
    extractor = CCCDPortraitExtractor(
        analyzer=analyzer,
        enable_rotation_retry=False,
    )
    card = np.full((630, 1000, 3), 128, dtype=np.uint8)

    try:
        extractor.extract(card)
    except FaceVerificationError as exc:
        assert exc.error_code == "CCCD_FACE_NOT_FOUND"
        assert exc.details["attempts"]
    else:
        raise AssertionError("CCCD không có chân dung phải bị từ chối.")

