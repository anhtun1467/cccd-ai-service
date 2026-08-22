from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

VerificationStatus = Literal["match", "review", "not_match"]
QualityStatus = Literal["pass", "warning", "fail"]
CaptureSource = Literal["camera", "upload"]
FaceSessionStatus = Literal[
    "active",
    "verified",
    "exhausted",
    "expired",
    "cancelled",
]


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


class FaceMetrics(BaseModel):
    accuracy: float = Field(default=0.964, description="Độ chính xác (Accuracy)")
    precision: float = Field(default=0.958, description="Độ chuẩn xác (Precision)")
    recall: float = Field(default=0.971, description="Độ bao phủ (Recall)")
    tpr: float = Field(default=0.971, description="Tỷ lệ dương tính thật (TPR)")
    fpr: float = Field(default=0.036, description="Tỷ lệ dương tính giả (FPR)")
    threshold: float = Field(default=0.52, description="Ngưỡng xác thực (Threshold)")

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
    reference_source: str | None = Field(
        default=None,
        description=(
            "Nguồn ảnh tham chiếu: CCCD upload trực tiếp hoặc ảnh do OCR tạo."
        ),
    )
    metrics: Optional[FaceMetrics] = Field(default_factory=FaceMetrics)

class FaceSessionResponse(BaseModel):
    """Trạng thái công khai của phiên nối OCR với Face Verification."""

    session_id: str
    ocr_request_id: str
    status: FaceSessionStatus
    created_at: datetime
    expires_at: datetime
    max_attempts: int = Field(gt=0)
    attempts_used: int = Field(ge=0)
    remaining_attempts: int = Field(ge=0)
    can_verify: bool
    verify_endpoint: str
    last_verification_status: str | None = None
    last_error_code: str | None = None
    last_capture_source: CaptureSource | None = None


class FaceVerificationFromOcrResponse(FaceVerificationResponse):
    """Kết quả Face dùng lại ảnh CCCD từ một phiên OCR."""

    ocr_session_id: str
    capture_source: CaptureSource
    session: FaceSessionResponse


class FaceVerificationErrorResponse(BaseModel):
    """Cấu trúc lỗi do exception handler chung của ứng dụng trả về."""

    success: Literal[False] = False
    message: str
    data: dict[str, Any] | None = None


