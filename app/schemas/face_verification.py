from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


VerificationStatus = Literal[
    "match",
    "review",
    "not_match",
]


class FaceVerificationResponse(BaseModel):
    """
    Dữ liệu trả về sau khi đối chiếu khuôn mặt
    giữa ảnh CCCD và ảnh webcam.
    """

    success: bool = Field(
        description="Cho biết pipeline có xử lý thành công hay không."
    )

    request_id: str = Field(
        description="Mã định danh duy nhất của lần xác minh."
    )

    status: VerificationStatus = Field(
        description="Kết quả MATCH, REVIEW hoặc NOT_MATCH."
    )

    is_match: bool = Field(
        description="True khi độ tương đồng đạt ngưỡng MATCH."
    )

    needs_review: bool = Field(
        description="True khi kết quả cần kiểm tra thủ công."
    )

    message: str = Field(
        description="Thông báo kết quả bằng tiếng Việt."
    )

    similarity: float = Field(
        ge=-1.0,
        le=1.0,
        description="Cosine similarity giữa hai khuôn mặt.",
    )

    distance: float = Field(
        ge=0.0,
        description="Khoảng cách được tính từ similarity.",
    )

    match_threshold: float = Field(
        ge=-1.0,
        le=1.0,
        description="Ngưỡng xác định khuôn mặt trùng khớp.",
    )

    review_threshold: float = Field(
        ge=-1.0,
        le=1.0,
        description="Ngưỡng xác định trường hợp cần kiểm tra.",
    )

    processing_time_ms: float = Field(
        ge=0.0,
        description="Thời gian xử lý tính bằng mili giây.",
    )

    cccd_detection_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Độ tin cậy phát hiện khuôn mặt trên CCCD.",
    )

    webcam_detection_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Độ tin cậy phát hiện khuôn mặt webcam.",
    )

    portrait_method: str = Field(
        description="Phương pháp trích xuất chân dung CCCD."
    )

    portrait_bbox: tuple[int, int, int, int] = Field(
        description="Tọa độ khuôn mặt trên ảnh CCCD."
    )


class FaceVerificationErrorResponse(BaseModel):
    """
    Dữ liệu lỗi chuẩn của API Face Verification.
    """

    success: bool = False
    detail: str
