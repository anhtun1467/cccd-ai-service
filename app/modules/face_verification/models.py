from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


VerificationStatus = Literal[
    "match",
    "review",
    "not_match",
]

FaceQualityStatus = Literal[
    "pass",
    "warning",
    "fail",
]

FaceImageKind = Literal[
    "cccd",
    "webcam",
]


@dataclass(frozen=True)
class FaceEmbeddingResult:
    """Khuôn mặt đã phát hiện cùng embedding nhận dạng."""

    embedding: np.ndarray
    bbox: np.ndarray
    detection_score: float
    landmarks: np.ndarray | None = None
    pose: np.ndarray | None = None

    @property
    def dimension(self) -> int:
        return int(self.embedding.reshape(-1).shape[0])

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


@dataclass(frozen=True)
class PortraitExtractionResult:
    """Chân dung được tách từ mặt trước CCCD."""

    portrait: np.ndarray
    bbox: tuple[int, int, int, int]
    detection_score: float
    extraction_method: str
    source_width: int
    source_height: int
    embedding_result: FaceEmbeddingResult | None = None
    detection_image: np.ndarray | None = None
    rotation_degrees: int = 0
    detected_face_count: int = 1


@dataclass(frozen=True)
class FaceQualityResult:
    """Kết quả đánh giá chất lượng một khuôn mặt."""

    source: FaceImageKind
    status: FaceQualityStatus
    is_acceptable: bool
    sharpness: float
    brightness: float
    face_width: int
    face_height: int
    face_height_ratio: float
    center_offset: float
    yaw: float | None
    pitch: float | None
    roll: float | None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def requires_review(self) -> bool:
        return self.status == "warning"

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "status": self.status,
            "is_acceptable": self.is_acceptable,
            "sharpness": round(self.sharpness, 2),
            "brightness": round(self.brightness, 2),
            "face_width": self.face_width,
            "face_height": self.face_height,
            "face_height_ratio": round(self.face_height_ratio, 4),
            "center_offset": round(self.center_offset, 4),
            "yaw": self._round_optional(self.yaw),
            "pitch": self._round_optional(self.pitch),
            "roll": self._round_optional(self.roll),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }

    @staticmethod
    def _round_optional(value: float | None) -> float | None:
        if value is None:
            return None
        return round(value, 2)

