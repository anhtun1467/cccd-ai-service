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
    D? li?u tr? v? sau khi d?i chi?u khuôn m?t
    gi?a ?nh CCCD và ?nh webcam.
    """

    success: bool = Field(
        description="Cho bi?t pipeline có x? lý thành công hay không."
    )

    request_id: str = Field(
        description="Mã d?nh danh duy nh?t c?a l?n xác minh."
    )

    status: VerificationStatus = Field(
        description="K?t qu? MATCH, REVIEW ho?c NOT_MATCH."
    )

    is_match: bool = Field(
        description="True khi d? tuong d?ng d?t ngu?ng MATCH."
    )

    needs_review: bool = Field(
        description="True khi k?t qu? c?n ki?m tra th? công."
    )

    message: str = Field(
        description="Thông báo k?t qu? b?ng ti?ng Vi?t."
    )

    similarity: float = Field(
        ge=-1.0,
        le=1.0,
        description="Cosine similarity gi?a hai khuôn m?t.",
    )

    distance: float = Field(
        ge=0.0,
        description="Kho?ng cách du?c tính t? similarity.",
    )

    match_threshold: float = Field(
        ge=-1.0,
        le=1.0,
        description="Ngu?ng xác d?nh khuôn m?t trùng kh?p.",
    )

    review_threshold: float = Field(
        ge=-1.0,
        le=1.0,
        description="Ngu?ng xác d?nh tru?ng h?p c?n ki?m tra.",
    )

    processing_time_ms: float = Field(
        ge=0.0,
        description="Th?i gian x? lý tính b?ng mili giây.",
    )

    cccd_detection_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Ð? tin c?y phát hi?n khuôn m?t trên CCCD.",
    )

    webcam_detection_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Ð? tin c?y phát hi?n khuôn m?t webcam.",
    )

    portrait_method: str = Field(
        description="Phuong pháp trích xu?t chân dung CCCD."
    )

    portrait_bbox: tuple[int, int, int, int] = Field(
        description="T?a d? khuôn m?t trên ?nh CCCD."
    )


class FaceVerificationErrorResponse(BaseModel):
    """
    D? li?u l?i chu?n c?a API Face Verification.
    """

    success: bool = False
    detail: str

