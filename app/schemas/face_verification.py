from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


VerificationStatus = Literal["match", "review", "not_match"]
QualityStatus = Literal["pass", "warning", "fail"]


class FaceQualityResponse(BaseModel):
    source: Literal["cccd", "webcam"]
    status: QualityStatus
    is_acceptable: bool
    sharpness: float = Field(ge=0.0)
    brightness: float = Field(ge=0.0, le=255.0)
    face_width: int = Field(gt=0)
    face_height: int = Field(gt=0)
    face_height_ratio: float = Field(gt=0.0)
    center_offset: float = Field(ge=0.0, le=1.0)
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class FaceVerificationResponse(BaseModel):
    """Kết quả đối chiếu khuôn mặt CCCD với ảnh webcam."""

    success: bool
    request_id: str
    status: VerificationStatus
    is_match: bool
    needs_review: bool
    message: str
    similarity: float = Field(ge=-1.0, le=1.0)
    distance: float = Field(ge=0.0, le=2.0)
    match_threshold: float = Field(ge=-1.0, le=1.0)
    review_threshold: float = Field(ge=-1.0, le=1.0)
    processing_time_ms: float = Field(ge=0.0)
    cccd_detection_score: float = Field(ge=0.0, le=1.0)
    webcam_detection_score: float = Field(ge=0.0, le=1.0)
    portrait_method: str
    portrait_bbox: tuple[int, int, int, int]
    portrait_rotation_degrees: int
    cccd_face_count: int = Field(ge=1)
    webcam_face_count: int = Field(ge=1)
    embedding_dimension: int = Field(gt=0)
    model_name: str
    quality_adjusted: bool
    liveness_checked: bool = Field(
        description=(
            "Luôn false với endpoint ảnh tĩnh; một ảnh không đủ chứng minh liveness."
        )
    )
    cccd_quality: FaceQualityResponse
    webcam_quality: FaceQualityResponse


class FaceVerificationErrorResponse(BaseModel):
    """Cấu trúc lỗi do exception handler chung của ứng dụng trả về."""

    success: Literal[False] = False
    message: str
    data: dict[str, Any] | None = None
