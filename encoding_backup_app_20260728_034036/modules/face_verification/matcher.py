from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FaceMatchResult:
    """
    Kết quả so khớp hai khuôn mặt.
    """

    is_match: bool
    similarity: float
    threshold: float

    @property
    def distance(self) -> float:
        return 1.0 - self.similarity


class CosineFaceMatcher:
    """
    So sánh hai embedding bằng cosine similarity.
    """

    def __init__(
        self,
        threshold: float = 0.45,
    ) -> None:
        if not -1.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold phải nằm trong khoảng -1 đến 1."
            )

        self.threshold = threshold

    def compare(
        self,
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
    ) -> FaceMatchResult:
        """
        So sánh hai vector embedding.
        """

        vector_a = self._prepare_embedding(
            embedding_a,
        )

        vector_b = self._prepare_embedding(
            embedding_b,
        )

        if vector_a.shape != vector_b.shape:
            raise ValueError(
                "Hai embedding phải có cùng số chiều."
            )

        similarity = float(
            np.dot(vector_a, vector_b)
        )

        similarity = float(
            np.clip(similarity, -1.0, 1.0)
        )

        return FaceMatchResult(
            is_match=similarity >= self.threshold,
            similarity=similarity,
            threshold=self.threshold,
        )

    @staticmethod
    def _prepare_embedding(
        embedding: np.ndarray,
    ) -> np.ndarray:
        if embedding is None:
            raise ValueError(
                "Embedding không được là None."
            )

        vector = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(-1)

        if vector.size == 0:
            raise ValueError("Embedding rỗng.")

        if not np.all(np.isfinite(vector)):
            raise ValueError(
                "Embedding chứa giá trị NaN hoặc Infinity."
            )

        norm = float(np.linalg.norm(vector))

        if norm <= 1e-12:
            raise ValueError(
                "Embedding có norm bằng 0."
            )

        return vector / norm
