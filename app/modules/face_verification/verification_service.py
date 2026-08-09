from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np

from app.modules.face_verification.errors import FaceVerificationError
from app.modules.face_verification.matcher import CosineFaceMatcher
from app.modules.face_verification.models import (
    FaceEmbeddingResult,
    FaceQualityResult,
    PortraitExtractionResult,
    VerificationStatus,
)


class PortraitExtractorProtocol(Protocol):
    def extract(self, card_image: np.ndarray) -> PortraitExtractionResult: ...


class FaceEmbedderProtocol(Protocol):
    model_name: str

    def extract(self, image: np.ndarray) -> list[FaceEmbeddingResult]: ...

    def extract_single(
        self,
        image: np.ndarray,
    ) -> FaceEmbeddingResult | None: ...


class QualityEvaluatorProtocol(Protocol):
    def evaluate(
        self,
        image: np.ndarray,
        face: FaceEmbeddingResult,
        source: str,
    ) -> FaceQualityResult: ...


@dataclass(frozen=True)
class FaceVerificationResult:
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
    portrait_rotation_degrees: int
    cccd_face_count: int
    webcam_face_count: int
    embedding_dimension: int
    model_name: str
    quality_adjusted: bool
    liveness_checked: bool
    cccd_quality: FaceQualityResult
    webcam_quality: FaceQualityResult

    @property
    def needs_review(self) -> bool:
        return self.status == "review"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "is_match": self.is_match,
            "needs_review": self.needs_review,
            "similarity": round(self.similarity, 4),
            "distance": round(self.distance, 4),
            "match_threshold": self.match_threshold,
            "review_threshold": self.review_threshold,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "cccd_detection_score": round(self.cccd_detection_score, 4),
            "webcam_detection_score": round(self.webcam_detection_score, 4),
            "portrait_method": self.portrait_method,
            "portrait_bbox": self.portrait_bbox,
            "portrait_rotation_degrees": self.portrait_rotation_degrees,
            "cccd_face_count": self.cccd_face_count,
            "webcam_face_count": self.webcam_face_count,
            "embedding_dimension": self.embedding_dimension,
            "model_name": self.model_name,
            "quality_adjusted": self.quality_adjusted,
            "liveness_checked": self.liveness_checked,
            "cccd_quality": self.cccd_quality.to_dict(),
            "webcam_quality": self.webcam_quality.to_dict(),
        }


@dataclass(frozen=True)
class FaceVerificationArtifacts:
    cccd_portrait: np.ndarray
    webcam_face: np.ndarray
    portrait_result: PortraitExtractionResult
    cccd_embedding_result: FaceEmbeddingResult
    webcam_embedding_result: FaceEmbeddingResult


@dataclass(frozen=True)
class FaceVerificationOutput:
    result: FaceVerificationResult
    artifacts: FaceVerificationArtifacts


class FaceVerificationService:
    """Đối chiếu chân dung CCCD với đúng một khuôn mặt webcam."""

    def __init__(
        self,
        portrait_extractor: PortraitExtractorProtocol | None = None,
        embedder: FaceEmbedderProtocol | None = None,
        quality_evaluator: QualityEvaluatorProtocol | None = None,
        match_threshold: float = 0.50,
        review_threshold: float = 0.40,
    ) -> None:
        if not -1.0 <= review_threshold <= 1.0:
            raise ValueError("review_threshold phải nằm trong khoảng -1 đến 1.")
        if not -1.0 <= match_threshold <= 1.0:
            raise ValueError("match_threshold phải nằm trong khoảng -1 đến 1.")
        if review_threshold >= match_threshold:
            raise ValueError("review_threshold phải nhỏ hơn match_threshold.")

        if embedder is None:
            from app.modules.face_verification.embedding import (
                InsightFaceEmbedder,
            )

            embedder = InsightFaceEmbedder()

        if portrait_extractor is None:
            from app.modules.face_verification.portrait_extractor import (
                CCCDPortraitExtractor,
            )

            portrait_extractor = CCCDPortraitExtractor(analyzer=embedder)

        if quality_evaluator is None:
            from app.modules.face_verification.quality import (
                FaceQualityEvaluator,
            )

            quality_evaluator = FaceQualityEvaluator()

        self.portrait_extractor = portrait_extractor
        self.embedder = embedder
        self.quality_evaluator = quality_evaluator
        self.match_threshold = match_threshold
        self.review_threshold = review_threshold
        self.matcher = CosineFaceMatcher(threshold=match_threshold)

    def verify(
        self,
        card_image: np.ndarray,
        webcam_image: np.ndarray,
    ) -> FaceVerificationOutput:
        self._validate_image(card_image, "card_image")
        self._validate_image(webcam_image, "webcam_image")
        start_time = perf_counter()

        portrait_result = self.portrait_extractor.extract(card_image)
        return self._verify_portrait_result(
            portrait_result=portrait_result,
            webcam_image=webcam_image,
            start_time=start_time,
        )

    def verify_prepared_portrait(
        self,
        portrait_image: np.ndarray,
        webcam_image: np.ndarray,
        *,
        extraction_method: str = "ocr_portrait_crop",
    ) -> FaceVerificationOutput:
        """Đối chiếu từ crop chân dung mà OCR đã chuẩn bị sẵn.

        Luồng này tránh phát hiện lại trên toàn bộ CCCD. Nếu crop OCR không
        đủ tốt, pipeline bên ngoài sẽ tự động quay về ảnh thẻ đã làm phẳng.
        """

        self._validate_image(portrait_image, "portrait_image")
        self._validate_image(webcam_image, "webcam_image")
        start_time = perf_counter()

        cccd_faces = self.embedder.extract(portrait_image)
        if not cccd_faces:
            raise FaceVerificationError(
                "CCCD_FACE_NOT_FOUND",
                "Không phát hiện được khuôn mặt trong crop chân dung của OCR.",
                details={
                    "referenceSource": extraction_method,
                    "suggestion": (
                        "Hệ thống sẽ thử lại trên ảnh CCCD đã làm phẳng."
                    ),
                },
            )

        cccd_face = max(
            cccd_faces,
            key=lambda face: (face.area, face.detection_score),
        )
        portrait_result = PortraitExtractionResult(
            portrait=self._crop_face(portrait_image, cccd_face),
            bbox=(
                cccd_face.x1,
                cccd_face.y1,
                cccd_face.x2,
                cccd_face.y2,
            ),
            detection_score=cccd_face.detection_score,
            extraction_method=extraction_method,
            source_width=int(portrait_image.shape[1]),
            source_height=int(portrait_image.shape[0]),
            embedding_result=cccd_face,
            detection_image=portrait_image,
            rotation_degrees=0,
            detected_face_count=len(cccd_faces),
        )
        return self._verify_portrait_result(
            portrait_result=portrait_result,
            webcam_image=webcam_image,
            start_time=start_time,
        )

    def _verify_portrait_result(
        self,
        *,
        portrait_result: PortraitExtractionResult,
        webcam_image: np.ndarray,
        start_time: float,
    ) -> FaceVerificationOutput:
        """Phần dùng chung sau khi đã xác định đúng chân dung CCCD."""

        cccd_embedding_result = portrait_result.embedding_result

        # Tương thích với extractor giả/cũ; extractor mới luôn trả embedding
        # ngay từ lần phát hiện đầu tiên và không chạy model lặp lại.
        if cccd_embedding_result is None:
            cccd_embedding_result = self.embedder.extract_single(
                portrait_result.portrait
            )
        if cccd_embedding_result is None:
            raise FaceVerificationError(
                "CCCD_EMBEDDING_NOT_FOUND",
                "Không trích xuất được đặc trưng khuôn mặt từ CCCD.",
            )

        webcam_faces = self.embedder.extract(webcam_image)
        if not webcam_faces:
            raise FaceVerificationError(
                "WEBCAM_FACE_NOT_FOUND",
                "Không phát hiện được khuôn mặt trong ảnh webcam.",
                details={
                    "suggestion": "Nhìn thẳng camera và chụp lại trong đủ sáng."
                },
            )
        if len(webcam_faces) != 1:
            raise FaceVerificationError(
                "MULTIPLE_WEBCAM_FACES",
                "Ảnh webcam phải có đúng một khuôn mặt.",
                details={
                    "faceCount": len(webcam_faces),
                    "suggestion": "Chỉ để một người xuất hiện trong khung hình.",
                },
            )

        webcam_embedding_result = webcam_faces[0]
        cccd_quality_image = (
            portrait_result.detection_image
            if portrait_result.detection_image is not None
            else portrait_result.portrait
        )
        cccd_quality = self.quality_evaluator.evaluate(
            cccd_quality_image,
            cccd_embedding_result,
            "cccd",
        )
        webcam_quality = self.quality_evaluator.evaluate(
            webcam_image,
            webcam_embedding_result,
            "webcam",
        )

        self._ensure_quality_is_acceptable(cccd_quality)
        self._ensure_quality_is_acceptable(webcam_quality)

        match_result = self.matcher.compare(
            cccd_embedding_result.embedding,
            webcam_embedding_result.embedding,
        )
        status = self._classify_similarity(match_result.similarity)

        quality_adjusted = False
        if status == "match" and (
            cccd_quality.requires_review
            or webcam_quality.requires_review
        ):
            status = "review"
            quality_adjusted = True

        processing_time_ms = (perf_counter() - start_time) * 1000.0
        result = FaceVerificationResult(
            status=status,
            is_match=status == "match",
            similarity=match_result.similarity,
            match_threshold=self.match_threshold,
            review_threshold=self.review_threshold,
            distance=match_result.distance,
            processing_time_ms=processing_time_ms,
            cccd_detection_score=cccd_embedding_result.detection_score,
            webcam_detection_score=webcam_embedding_result.detection_score,
            portrait_method=portrait_result.extraction_method,
            portrait_bbox=portrait_result.bbox,
            portrait_rotation_degrees=portrait_result.rotation_degrees,
            cccd_face_count=portrait_result.detected_face_count,
            webcam_face_count=len(webcam_faces),
            embedding_dimension=cccd_embedding_result.dimension,
            model_name=getattr(self.embedder, "model_name", "unknown"),
            quality_adjusted=quality_adjusted,
            # Một ảnh tĩnh không đủ để khẳng định liveness.
            liveness_checked=False,
            cccd_quality=cccd_quality,
            webcam_quality=webcam_quality,
        )
        artifacts = FaceVerificationArtifacts(
            cccd_portrait=portrait_result.portrait,
            webcam_face=self._crop_face(webcam_image, webcam_embedding_result),
            portrait_result=portrait_result,
            cccd_embedding_result=cccd_embedding_result,
            webcam_embedding_result=webcam_embedding_result,
        )
        return FaceVerificationOutput(result=result, artifacts=artifacts)

    def _classify_similarity(self, similarity: float) -> VerificationStatus:
        if similarity >= self.match_threshold:
            return "match"
        if similarity >= self.review_threshold:
            return "review"
        return "not_match"

    @staticmethod
    def _ensure_quality_is_acceptable(quality: FaceQualityResult) -> None:
        if quality.is_acceptable:
            return
        first_error = quality.errors[0] if quality.errors else "FACE_QUALITY_FAILED"
        source_name = "CCCD" if quality.source == "cccd" else "webcam"
        raise FaceVerificationError(
            first_error,
            f"Chất lượng khuôn mặt {source_name} không đạt yêu cầu.",
            details={
                "quality": quality.to_dict(),
                "suggestion": (
                    "Chụp lại ảnh rõ nét, đủ sáng, nhìn thẳng và không che mặt."
                ),
            },
        )

    @staticmethod
    def _crop_face(
        image: np.ndarray,
        face: FaceEmbeddingResult,
        margin_ratio: float = 0.18,
    ) -> np.ndarray:
        image_height, image_width = image.shape[:2]
        margin_x = int(face.width * margin_ratio)
        margin_y = int(face.height * margin_ratio)
        x1 = max(0, face.x1 - margin_x)
        y1 = max(0, face.y1 - margin_y)
        x2 = min(image_width, face.x2 + margin_x)
        y2 = min(image_height, face.y2 + margin_y)
        if x2 <= x1 or y2 <= y1:
            return image.copy()
        return image[y1:y2, x1:x2].copy()

    @staticmethod
    def _validate_image(image: np.ndarray, image_name: str) -> None:
        if image is None:
            raise ValueError(f"{image_name} không được là None.")
        if not isinstance(image, np.ndarray):
            raise TypeError(f"{image_name} phải là numpy.ndarray.")
        if image.size == 0:
            raise ValueError(f"{image_name} không được rỗng.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{image_name} phải là ảnh BGR 3 kênh.")
