from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Sequence

import cv2
import numpy as np

from app.modules.face_verification.errors import FaceVerificationError


@dataclass(frozen=True)
class DetectedFace:
    bbox: np.ndarray
    score: float
    landmarks: np.ndarray | None = None
    pose: np.ndarray | None = None

    @property
    def x1(self) -> int:
        return int(round(float(self.bbox[0])))

    @property
    def y1(self) -> int:
        return int(round(float(self.bbox[1])))

    @property
    def x2(self) -> int:
        return int(round(float(self.bbox[2])))

    @property
    def y2(self) -> int:
        return int(round(float(self.bbox[3])))

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height


class InsightFaceDetector:
    """Detector InsightFace độc lập dành cho màn hình preview camera."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        detection_size: tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.60,
        providers: Sequence[str] | None = None,
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
        self._face_analysis: Any | None = None
        self._load_lock = Lock()

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        self._validate_image(image)
        raw_faces = self._get_face_analysis().get(image)
        faces: list[DetectedFace] = []

        for face in raw_faces:
            score = float(getattr(face, "det_score", 0.0))
            if score < self.confidence_threshold:
                continue
            bbox = np.asarray(face.bbox, dtype=np.float32).reshape(-1)
            if bbox.size < 4:
                continue
            landmarks = self._optional_array(getattr(face, "kps", None))
            pose = self._optional_array(getattr(face, "pose", None))
            faces.append(
                DetectedFace(
                    bbox=bbox[:4],
                    score=score,
                    landmarks=landmarks,
                    pose=pose,
                )
            )

        faces.sort(key=lambda face: (face.area, face.score), reverse=True)
        return faces

    def detect_single(self, image: np.ndarray) -> DetectedFace | None:
        faces = self.detect(image)
        return faces[0] if faces else None

    @staticmethod
    def crop_face(
        image: np.ndarray,
        face: DetectedFace,
        margin_ratio: float = 0.15,
    ) -> np.ndarray:
        if margin_ratio < 0:
            raise ValueError("margin_ratio không được nhỏ hơn 0.")
        image_height, image_width = image.shape[:2]
        margin_x = int(face.width * margin_ratio)
        margin_y = int(face.height * margin_ratio)
        x1 = max(0, face.x1 - margin_x)
        y1 = max(0, face.y1 - margin_y)
        x2 = min(image_width, face.x2 + margin_x)
        y2 = min(image_height, face.y2 + margin_y)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Bounding box khuôn mặt không hợp lệ.")
        return image[y1:y2, x1:x2].copy()

    @staticmethod
    def draw_faces(
        image: np.ndarray,
        faces: Sequence[DetectedFace],
    ) -> np.ndarray:
        output = image.copy()
        for index, face in enumerate(faces, start=1):
            cv2.rectangle(
                output,
                (face.x1, face.y1),
                (face.x2, face.y2),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                output,
                f"Face {index}: {face.score:.3f}",
                (face.x1, max(25, face.y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            if face.landmarks is not None:
                for point in face.landmarks:
                    cv2.circle(
                        output,
                        (int(point[0]), int(point[1])),
                        3,
                        (0, 0, 255),
                        -1,
                    )
        return output

    def _get_face_analysis(self) -> Any:
        if self._face_analysis is not None:
            return self._face_analysis
        with self._load_lock:
            if self._face_analysis is not None:
                return self._face_analysis
            try:
                from insightface.app import FaceAnalysis

                analysis = FaceAnalysis(
                    name=self.model_name,
                    allowed_modules=["detection"],
                    providers=self.providers,
                )
                uses_cuda = any(
                    provider.startswith("CUDA")
                    for provider in self.providers
                )
                analysis.prepare(
                    ctx_id=0 if uses_cuda else -1,
                    det_size=self.detection_size,
                )
            except Exception as exc:  # pragma: no cover - phụ thuộc máy chạy
                raise FaceVerificationError(
                    "FACE_DETECTOR_UNAVAILABLE",
                    "Không thể nạp detector InsightFace.",
                    status_code=503,
                    details={"reason": str(exc)},
                ) from exc
            self._face_analysis = analysis
            return self._face_analysis

    @staticmethod
    def _optional_array(value: object) -> np.ndarray | None:
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float32)
        return array if array.size else None

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if image is None or not isinstance(image, np.ndarray):
            raise TypeError("Ảnh đầu vào phải là numpy.ndarray.")
        if image.size == 0:
            raise ValueError("Ảnh đầu vào bị rỗng.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Ảnh đầu vào phải là ảnh BGR 3 kênh.")
