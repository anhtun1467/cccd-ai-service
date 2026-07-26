from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np

from app.modules.face_verification.embedding import (
    FaceEmbeddingResult,
    InsightFaceEmbedder,
)
from app.modules.face_verification.matcher import (
    CosineFaceMatcher,
)
from app.modules.face_verification.portrait_extractor import (
    CCCDPortraitExtractor,
    PortraitExtractionResult,
)


VerificationStatus = Literal[
    "match",
    "not_match",
    "review",
]


@dataclass(frozen=True)
class FaceVerificationResult:
    """
    Kết quả xác minh khuôn mặt CCCD với ảnh webcam.
    """

    status: VerificationStatus
    is_match: bool
    similarity: float
    match_threshold: float
    review_threshold: float
    distance: float
    processing_time_ms: float

    cccd_detection_score: float
    webcam_detection_score: float

    portrait_method: str
    portrait_bbox: tuple[int, int, int, int]

    @property
    def needs_review(self) -> bool:
        return self.status == "review"

    def to_dict(self) -> dict[str, object]:
        """
        Chuyển kết quả thành dictionary để trả về API.
        """

        return {
            "status": self.status,
            "is_match": self.is_match,
            "needs_review": self.needs_review,
            "similarity": round(self.similarity, 4),
            "distance": round(self.distance, 4),
            "match_threshold": self.match_threshold,
            "review_threshold": self.review_threshold,
            "processing_time_ms": round(
                self.processing_time_ms,
                2,
            ),
            "cccd_detection_score": round(
                self.cccd_detection_score,
                4,
            ),
            "webcam_detection_score": round(
                self.webcam_detection_score,
                4,
            ),
            "portrait_method": self.portrait_method,
            "portrait_bbox": self.portrait_bbox,
        }


@dataclass(frozen=True)
class FaceVerificationArtifacts:
    """
    Dữ liệu trung gian phục vụ debug và lưu ảnh.
    """

    cccd_portrait: np.ndarray
    portrait_result: PortraitExtractionResult
    cccd_embedding_result: FaceEmbeddingResult
    webcam_embedding_result: FaceEmbeddingResult


@dataclass(frozen=True)
class FaceVerificationOutput:
    """
    Kết quả đầy đủ gồm dữ liệu nghiệp vụ và artifacts.
    """

    result: FaceVerificationResult
    artifacts: FaceVerificationArtifacts


class FaceVerificationService:
    """
    Service xác minh khuôn mặt giữa CCCD và ảnh webcam.

    Luồng xử lý:
    1. Trích xuất chân dung từ ảnh CCCD.
    2. Trích xuất embedding chân dung CCCD.
    3. Trích xuất embedding khuôn mặt webcam.
    4. Tính cosine similarity.
    5. Phân loại MATCH / REVIEW / NOT_MATCH.
    """

    def __init__(
        self,
        portrait_extractor: CCCDPortraitExtractor | None = None,
        embedder: InsightFaceEmbedder | None = None,
        match_threshold: float = 0.50,
        review_threshold: float = 0.40,
    ) -> None:
        if not -1.0 <= review_threshold <= 1.0:
            raise ValueError(
                "review_threshold phải nằm trong khoảng -1 đến 1."
            )

        if not -1.0 <= match_threshold <= 1.0:
            raise ValueError(
                "match_threshold phải nằm trong khoảng -1 đến 1."
            )

        if review_threshold >= match_threshold:
            raise ValueError(
                "review_threshold phải nhỏ hơn match_threshold."
            )

        self.portrait_extractor = (
            portrait_extractor
            or CCCDPortraitExtractor()
        )

        self.embedder = (
            embedder
            or InsightFaceEmbedder(
                model_name="buffalo_l",
                detection_size=(640, 640),
                confidence_threshold=0.50,
                providers=["CPUExecutionProvider"],
            )
        )

        self.match_threshold = match_threshold
        self.review_threshold = review_threshold

        self.matcher = CosineFaceMatcher(
            threshold=match_threshold,
        )

    def verify(
        self,
        card_image: np.ndarray,
        webcam_image: np.ndarray,
    ) -> FaceVerificationOutput:
        """
        Xác minh khuôn mặt giữa ảnh CCCD và ảnh webcam.

        Args:
            card_image:
                Ảnh mặt trước CCCD, định dạng BGR.

            webcam_image:
                Ảnh khuôn mặt chụp từ webcam, định dạng BGR.

        Returns:
            FaceVerificationOutput.

        Raises:
            ValueError:
                Khi ảnh đầu vào không hợp lệ.

            RuntimeError:
                Khi không phát hiện hoặc không trích xuất được
                embedding khuôn mặt.
        """

        self._validate_image(
            card_image,
            image_name="card_image",
        )

        self._validate_image(
            webcam_image,
            image_name="webcam_image",
        )

        start_time = perf_counter()

        portrait_result = self.portrait_extractor.extract(
            card_image
        )

        cccd_embedding_result = self.embedder.extract_single(
            portrait_result.portrait
        )

        if cccd_embedding_result is None:
            raise RuntimeError(
                "Không thể trích xuất embedding từ "
                "chân dung CCCD."
            )

        webcam_embedding_result = self.embedder.extract_single(
            webcam_image
        )

        if webcam_embedding_result is None:
            raise RuntimeError(
                "Không phát hiện được khuôn mặt "
                "trong ảnh webcam."
            )

        match_result = self.matcher.compare(
            cccd_embedding_result.embedding,
            webcam_embedding_result.embedding,
        )

        status = self._classify_similarity(
            match_result.similarity
        )

        processing_time_ms = (
            perf_counter() - start_time
        ) * 1000.0

        result = FaceVerificationResult(
            status=status,
            is_match=status == "match",
            similarity=match_result.similarity,
            match_threshold=self.match_threshold,
            review_threshold=self.review_threshold,
            distance=match_result.distance,
            processing_time_ms=processing_time_ms,
            cccd_detection_score=(
                cccd_embedding_result.detection_score
            ),
            webcam_detection_score=(
                webcam_embedding_result.detection_score
            ),
            portrait_method=(
                portrait_result.extraction_method
            ),
            portrait_bbox=portrait_result.bbox,
        )

        artifacts = FaceVerificationArtifacts(
            cccd_portrait=portrait_result.portrait,
            portrait_result=portrait_result,
            cccd_embedding_result=(
                cccd_embedding_result
            ),
            webcam_embedding_result=(
                webcam_embedding_result
            ),
        )

        return FaceVerificationOutput(
            result=result,
            artifacts=artifacts,
        )

    def _classify_similarity(
        self,
        similarity: float,
    ) -> VerificationStatus:
        """
        Phân loại similarity thành ba mức.
        """

        if similarity >= self.match_threshold:
            return "match"

        if similarity >= self.review_threshold:
            return "review"

        return "not_match"

    @staticmethod
    def _validate_image(
        image: np.ndarray,
        image_name: str,
    ) -> None:
        if image is None:
            raise ValueError(
                f"{image_name} không được là None."
            )

        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"{image_name} phải là numpy.ndarray."
            )

        if image.size == 0:
            raise ValueError(
                f"{image_name} không được rỗng."
            )

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"{image_name} phải là ảnh BGR 3 kênh."
            )
