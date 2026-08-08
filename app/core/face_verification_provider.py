from __future__ import annotations

from threading import Lock

from app.core.config import Settings, settings
from app.modules.face_verification.embedding import InsightFaceEmbedder
from app.modules.face_verification.portrait_extractor import (
    CCCDPortraitExtractor,
)
from app.modules.face_verification.quality import FaceQualityEvaluator
from app.modules.face_verification.verification_service import (
    FaceVerificationService,
)


class FaceVerificationProvider:
    """Singleton nạp đúng một phiên InsightFace cho toàn bộ API."""

    _instance: "FaceVerificationProvider | None" = None
    _instance_lock = Lock()

    def __init__(self, app_settings: Settings | None = None) -> None:
        self.settings = app_settings or settings
        self._service: FaceVerificationService | None = None
        self._embedder: InsightFaceEmbedder | None = None
        self._portrait_extractor: CCCDPortraitExtractor | None = None
        self._service_lock = Lock()

    @classmethod
    def instance(cls) -> "FaceVerificationProvider":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def service(self) -> FaceVerificationService:
        if self._service is None:
            with self._service_lock:
                if self._service is None:
                    self._build_service()
        assert self._service is not None
        return self._service

    @property
    def embedder(self) -> InsightFaceEmbedder:
        _ = self.service
        assert self._embedder is not None
        return self._embedder

    @property
    def portrait_extractor(self) -> CCCDPortraitExtractor:
        _ = self.service
        assert self._portrait_extractor is not None
        return self._portrait_extractor

    def warmup(self) -> None:
        self.embedder.warmup()

    def _build_service(self) -> None:
        detection_size = self.settings.face_detection_size
        embedder = InsightFaceEmbedder(
            model_name=self.settings.face_model_name,
            detection_size=(detection_size, detection_size),
            confidence_threshold=self.settings.face_detection_confidence,
            providers=[self.settings.face_execution_provider],
        )
        portrait_extractor = CCCDPortraitExtractor(analyzer=embedder)
        service = FaceVerificationService(
            portrait_extractor=portrait_extractor,
            embedder=embedder,
            quality_evaluator=FaceQualityEvaluator(),
            match_threshold=self.settings.face_match_threshold,
            review_threshold=self.settings.face_review_threshold,
        )

        self._embedder = embedder
        self._portrait_extractor = portrait_extractor
        self._service = service

    @classmethod
    def reset_for_tests(cls) -> None:
        """Xóa singleton trong unit test; không dùng trong request thật."""

        with cls._instance_lock:
            cls._instance = None
