from __future__ import annotations

from threading import Lock

from app.modules.face_verification.embedding import (
    InsightFaceEmbedder,
)
from app.modules.face_verification.portrait_extractor import (
    CCCDPortraitExtractor,
)
from app.modules.face_verification.verification_service import (
    FaceVerificationService,
)


class FaceVerificationProvider:
    """
    Singleton Provider qu?n lư toàn b? tài nguyên Face Verification.

    M?c tiêu:
        - Ch? load InsightFace m?t l?n.
        - Tái s? d?ng model cho m?i request.
        - Tránh m?t th?i gian kh?i t?o model liên t?c.
    """

    _instance: "FaceVerificationProvider | None" = None
    _lock = Lock()

    def __init__(self) -> None:
        self._portrait_extractor = CCCDPortraitExtractor()

        self._embedder = InsightFaceEmbedder(
            model_name="buffalo_l",
            detection_size=(640, 640),
            confidence_threshold=0.50,
            providers=["CPUExecutionProvider"],
        )

        self._service = FaceVerificationService(
            portrait_extractor=self._portrait_extractor,
            embedder=self._embedder,
            match_threshold=0.50,
            review_threshold=0.40,
        )

    @classmethod
    def instance(cls) -> "FaceVerificationProvider":
        """
        L?y singleton provider.
        """

        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()

        return cls._instance

    @property
    def portrait_extractor(self) -> CCCDPortraitExtractor:
        return self._portrait_extractor

    @property
    def embedder(self) -> InsightFaceEmbedder:
        return self._embedder

    @property
    def service(self) -> FaceVerificationService:
        return self._service


provider = FaceVerificationProvider.instance()
