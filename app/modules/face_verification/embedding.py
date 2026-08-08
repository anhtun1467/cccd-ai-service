from __future__ import annotations

from threading import Lock
from typing import Any, Sequence

import numpy as np

from app.modules.face_verification.errors import FaceVerificationError
from app.modules.face_verification.models import FaceEmbeddingResult


class InsightFaceEmbedder:
    """Phát hiện, căn chỉnh và sinh embedding bằng InsightFace.

    Model được nạp lười ở lần sử dụng đầu tiên. Cách này giúp API OCR vẫn
    khởi động bình thường khi model Face chưa được cài hoặc chưa tải xong.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        detection_size: tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.50,
        providers: Sequence[str] | None = None,
        model_root: str | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold phải nằm trong khoảng từ 0 đến 1."
            )
        if detection_size[0] <= 0 or detection_size[1] <= 0:
            raise ValueError("detection_size phải lớn hơn 0.")

        self.model_name = model_name
        self.detection_size = detection_size
        self.confidence_threshold = confidence_threshold
        self.providers = list(providers or ["CPUExecutionProvider"])
        self.model_root = model_root

        self._face_analysis: Any | None = None
        self._load_lock = Lock()

    @property
    def is_loaded(self) -> bool:
        return self._face_analysis is not None

    def warmup(self) -> None:
        """Nạp model chủ động khi cần kiểm tra readiness."""

        self._get_face_analysis()

    def extract(self, image: np.ndarray) -> list[FaceEmbeddingResult]:
        """Trả về toàn bộ khuôn mặt đạt ngưỡng phát hiện."""

        self._validate_image(image)
        face_analysis = self._get_face_analysis()

        try:
            raw_faces = face_analysis.get(image)
        except Exception as exc:  # pragma: no cover - phụ thuộc ONNX Runtime
            raise FaceVerificationError(
                "FACE_MODEL_INFERENCE_FAILED",
                "InsightFace không thể xử lý ảnh đầu vào.",
                status_code=503,
                details={"reason": str(exc)},
            ) from exc

        image_height, image_width = image.shape[:2]
        results: list[FaceEmbeddingResult] = []

        for face in raw_faces:
            detection_score = float(getattr(face, "det_score", 0.0))
            if detection_score < self.confidence_threshold:
                continue

            raw_embedding = getattr(face, "normed_embedding", None)
            if raw_embedding is None:
                raw_embedding = getattr(face, "embedding", None)
            if raw_embedding is None:
                continue

            embedding = self.normalize_embedding(
                np.asarray(raw_embedding, dtype=np.float32)
            )
            if not np.all(np.isfinite(embedding)):
                continue

            bbox = np.asarray(face.bbox, dtype=np.float32).reshape(-1)
            if bbox.size < 4:
                continue

            bbox = np.asarray(
                [
                    np.clip(bbox[0], 0, max(0, image_width - 1)),
                    np.clip(bbox[1], 0, max(0, image_height - 1)),
                    np.clip(bbox[2], 1, image_width),
                    np.clip(bbox[3], 1, image_height),
                ],
                dtype=np.float32,
            )
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue

            landmarks = self._optional_array(getattr(face, "kps", None))
            pose = self._optional_array(getattr(face, "pose", None))

            results.append(
                FaceEmbeddingResult(
                    embedding=embedding,
                    bbox=bbox,
                    detection_score=detection_score,
                    landmarks=landmarks,
                    pose=pose,
                )
            )

        results.sort(
            key=lambda result: (result.area, result.detection_score),
            reverse=True,
        )
        return results

    def extract_single(
        self,
        image: np.ndarray,
    ) -> FaceEmbeddingResult | None:
        """Trả về khuôn mặt lớn nhất; chỉ dùng ở luồng tương thích cũ."""

        results = self.extract(image)
        return results[0] if results else None

    @staticmethod
    def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.size == 0:
            raise ValueError("Embedding bị rỗng.")
        if not np.all(np.isfinite(vector)):
            raise ValueError("Embedding chứa NaN hoặc Infinity.")

        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise ValueError("Embedding không hợp lệ vì có norm bằng 0.")
        return vector / norm

    def _get_face_analysis(self) -> Any:
        if self._face_analysis is not None:
            return self._face_analysis

        with self._load_lock:
            if self._face_analysis is not None:
                return self._face_analysis

            try:
                from insightface.app import FaceAnalysis

                kwargs: dict[str, object] = {
                    "name": self.model_name,
                    "allowed_modules": [
                        "detection",
                        "recognition",
                        "landmark_3d_68",
                    ],
                    "providers": self.providers,
                }
                if self.model_root:
                    kwargs["root"] = self.model_root

                face_analysis = FaceAnalysis(**kwargs)
                uses_cuda = any(
                    provider.startswith("CUDA")
                    for provider in self.providers
                )
                face_analysis.prepare(
                    ctx_id=0 if uses_cuda else -1,
                    det_size=self.detection_size,
                )
            except Exception as exc:  # pragma: no cover - phụ thuộc máy chạy
                raise FaceVerificationError(
                    "FACE_MODEL_UNAVAILABLE",
                    "Không thể nạp model InsightFace buffalo_l.",
                    status_code=503,
                    details={
                        "model": self.model_name,
                        "providers": self.providers,
                        "reason": str(exc),
                    },
                ) from exc

            self._face_analysis = face_analysis
            return self._face_analysis

    @staticmethod
    def _optional_array(value: object) -> np.ndarray | None:
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float32)
        if array.size == 0 or not np.all(np.isfinite(array)):
            return None
        return array

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if image is None:
            raise ValueError("Ảnh đầu vào không được là None.")
        if not isinstance(image, np.ndarray):
            raise TypeError("Ảnh đầu vào phải là numpy.ndarray.")
        if image.size == 0:
            raise ValueError("Ảnh đầu vào bị rỗng.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Ảnh đầu vào phải là ảnh BGR 3 kênh.")


__all__ = [
    "FaceEmbeddingResult",
    "InsightFaceEmbedder",
]
