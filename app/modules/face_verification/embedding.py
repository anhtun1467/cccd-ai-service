from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from insightface.app import FaceAnalysis


@dataclass(frozen=True)
class FaceEmbeddingResult:
    """
    Kết quả trích xuất đặc trưng khuôn mặt.
    """

    embedding: np.ndarray
    bbox: np.ndarray
    detection_score: float
    landmarks: np.ndarray | None

    @property
    def dimension(self) -> int:
        return int(self.embedding.shape[0])


class InsightFaceEmbedder:
    """
    Phát hiện và trích xuất embedding khuôn mặt bằng InsightFace.

    Model buffalo_l bao gồm:
    - Face detection
    - Face landmarks
    - Face recognition embedding
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        detection_size: tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.60,
        providers: Sequence[str] | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold phải nằm trong khoảng 0 đến 1."
            )

        self.model_name = model_name
        self.detection_size = detection_size
        self.confidence_threshold = confidence_threshold
        self.providers = list(
            providers or ["CPUExecutionProvider"]
        )

        self._face_analysis = FaceAnalysis(
            name=self.model_name,
            allowed_modules=[
                "detection",
                "recognition",
            ],
            providers=self.providers,
        )

        self._face_analysis.prepare(
            ctx_id=-1,
            det_size=self.detection_size,
        )

    def extract(
        self,
        image: np.ndarray,
    ) -> list[FaceEmbeddingResult]:
        """
        Trích xuất embedding của tất cả khuôn mặt hợp lệ trong ảnh.
        """

        self._validate_image(image)

        raw_faces = self._face_analysis.get(image)

        results: list[FaceEmbeddingResult] = []

        for face in raw_faces:
            detection_score = float(face.det_score)

            if detection_score < self.confidence_threshold:
                continue

            raw_embedding = getattr(face, "embedding", None)

            if raw_embedding is None:
                continue

            normalized_embedding = self.normalize_embedding(
                np.asarray(
                    raw_embedding,
                    dtype=np.float32,
                )
            )

            bbox = np.asarray(
                face.bbox,
                dtype=np.float32,
            )

            landmarks = None

            if getattr(face, "kps", None) is not None:
                landmarks = np.asarray(
                    face.kps,
                    dtype=np.float32,
                )

            results.append(
                FaceEmbeddingResult(
                    embedding=normalized_embedding,
                    bbox=bbox,
                    detection_score=detection_score,
                    landmarks=landmarks,
                )
            )

        results.sort(
            key=self._face_area,
            reverse=True,
        )

        return results

    def extract_single(
        self,
        image: np.ndarray,
    ) -> FaceEmbeddingResult | None:
        """
        Trả về embedding của khuôn mặt lớn nhất.

        Nếu không tìm thấy khuôn mặt hợp lệ thì trả về None.
        """

        results = self.extract(image)

        if not results:
            return None

        return results[0]

    @staticmethod
    def normalize_embedding(
        embedding: np.ndarray,
    ) -> np.ndarray:
        """
        Chuẩn hóa vector theo L2 norm.
        """

        if embedding.ndim != 1:
            embedding = embedding.reshape(-1)

        norm = float(np.linalg.norm(embedding))

        if norm <= 1e-12:
            raise ValueError(
                "Embedding không hợp lệ vì có norm bằng 0."
            )

        return embedding / norm

    @staticmethod
    def _face_area(
        result: FaceEmbeddingResult,
    ) -> float:
        x1, y1, x2, y2 = result.bbox

        width = max(0.0, float(x2 - x1))
        height = max(0.0, float(y2 - y1))

        return width * height

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if image is None:
            raise ValueError("Ảnh đầu vào không được là None.")

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "Ảnh đầu vào phải là numpy.ndarray."
            )

        if image.size == 0:
            raise ValueError("Ảnh đầu vào rỗng.")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "Ảnh đầu vào phải là ảnh BGR 3 kênh."
            )