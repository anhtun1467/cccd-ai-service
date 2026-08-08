from __future__ import annotations

import unittest

import numpy as np

from app.modules.face_verification.errors import FaceVerificationError
from app.modules.face_verification.models import (
    FaceEmbeddingResult,
    FaceQualityResult,
    PortraitExtractionResult,
)
from app.modules.face_verification.verification_service import (
    FaceVerificationService,
)


def make_face(
    embedding: np.ndarray,
    *,
    bbox: tuple[int, int, int, int],
    score: float = 0.95,
) -> FaceEmbeddingResult:
    return FaceEmbeddingResult(
        embedding=np.asarray(embedding, dtype=np.float32),
        bbox=np.asarray(bbox, dtype=np.float32),
        detection_score=score,
        landmarks=np.asarray(
            [[30, 40], [70, 40], [50, 60], [35, 80], [65, 80]],
            dtype=np.float32,
        ),
    )


def make_quality(
    source: str,
    *,
    status: str = "pass",
    errors: tuple[str, ...] = (),
    warnings: tuple[str, ...] = (),
) -> FaceQualityResult:
    return FaceQualityResult(
        source=source,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        is_acceptable=not errors,
        sharpness=120.0,
        brightness=128.0,
        face_width=120,
        face_height=150,
        face_height_ratio=0.40,
        center_offset=0.05,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        errors=errors,
        warnings=warnings,
    )


class FakePortraitExtractor:
    def __init__(self, face: FaceEmbeddingResult) -> None:
        self.face = face

    def extract(self, card_image: np.ndarray) -> PortraitExtractionResult:
        portrait = card_image[20:220, 10:170].copy()
        return PortraitExtractionResult(
            portrait=portrait,
            bbox=(10, 20, 170, 220),
            detection_score=self.face.detection_score,
            extraction_method="fake_portrait_extractor",
            source_width=card_image.shape[1],
            source_height=card_image.shape[0],
            embedding_result=self.face,
            detection_image=card_image,
            rotation_degrees=0,
            detected_face_count=1,
        )


class FakeEmbedder:
    model_name = "buffalo_l-test"

    def __init__(self, webcam_faces: list[FaceEmbeddingResult]) -> None:
        self.webcam_faces = webcam_faces

    def extract(self, image: np.ndarray) -> list[FaceEmbeddingResult]:
        return list(self.webcam_faces)

    def extract_single(self, image: np.ndarray) -> FaceEmbeddingResult | None:
        return self.webcam_faces[0] if self.webcam_faces else None


class FakeQualityEvaluator:
    def __init__(
        self,
        cccd_quality: FaceQualityResult | None = None,
        webcam_quality: FaceQualityResult | None = None,
    ) -> None:
        self.cccd_quality = cccd_quality or make_quality("cccd")
        self.webcam_quality = webcam_quality or make_quality("webcam")

    def evaluate(
        self,
        image: np.ndarray,
        face: FaceEmbeddingResult,
        source: str,
    ) -> FaceQualityResult:
        return self.cccd_quality if source == "cccd" else self.webcam_quality


class FaceVerificationServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.card_image = np.zeros((630, 1000, 3), dtype=np.uint8)
        self.webcam_image = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.base_embedding = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        self.card_face = make_face(
            self.base_embedding,
            bbox=(50, 170, 190, 360),
            score=0.94,
        )

    def create_service(
        self,
        webcam_embeddings: list[np.ndarray],
        *,
        cccd_quality: FaceQualityResult | None = None,
        webcam_quality: FaceQualityResult | None = None,
        match_threshold: float = 0.50,
        review_threshold: float = 0.40,
    ) -> FaceVerificationService:
        webcam_faces = [
            make_face(
                embedding,
                bbox=(420 + index * 20, 160, 820 + index * 20, 650),
                score=0.90,
            )
            for index, embedding in enumerate(webcam_embeddings)
        ]
        return FaceVerificationService(
            portrait_extractor=FakePortraitExtractor(self.card_face),
            embedder=FakeEmbedder(webcam_faces),
            quality_evaluator=FakeQualityEvaluator(
                cccd_quality=cccd_quality,
                webcam_quality=webcam_quality,
            ),
            match_threshold=match_threshold,
            review_threshold=review_threshold,
        )

    def test_match_status(self) -> None:
        service = self.create_service([self.base_embedding])
        output = service.verify(self.card_image, self.webcam_image)

        self.assertEqual(output.result.status, "match")
        self.assertTrue(output.result.is_match)
        self.assertFalse(output.result.needs_review)
        self.assertAlmostEqual(output.result.similarity, 1.0, places=5)
        self.assertEqual(output.result.embedding_dimension, 3)
        self.assertEqual(output.result.webcam_face_count, 1)

    def test_review_status(self) -> None:
        similarity = 0.45
        embedding = np.asarray(
            [similarity, np.sqrt(1.0 - similarity**2), 0.0],
            dtype=np.float32,
        )
        output = self.create_service([embedding]).verify(
            self.card_image,
            self.webcam_image,
        )
        self.assertEqual(output.result.status, "review")
        self.assertTrue(output.result.needs_review)
        self.assertAlmostEqual(output.result.similarity, 0.45, places=4)

    def test_not_match_status(self) -> None:
        embedding = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        output = self.create_service([embedding]).verify(
            self.card_image,
            self.webcam_image,
        )
        self.assertEqual(output.result.status, "not_match")
        self.assertFalse(output.result.is_match)

    def test_quality_warning_downgrades_match_to_review(self) -> None:
        webcam_quality = make_quality(
            "webcam",
            status="warning",
            warnings=("WEBCAM_FACE_SLIGHTLY_BLURRY",),
        )
        output = self.create_service(
            [self.base_embedding],
            webcam_quality=webcam_quality,
        ).verify(self.card_image, self.webcam_image)

        self.assertEqual(output.result.status, "review")
        self.assertTrue(output.result.quality_adjusted)
        self.assertFalse(output.result.is_match)

    def test_multiple_webcam_faces_are_rejected(self) -> None:
        service = self.create_service(
            [self.base_embedding, self.base_embedding]
        )
        with self.assertRaises(FaceVerificationError) as context:
            service.verify(self.card_image, self.webcam_image)
        self.assertEqual(context.exception.error_code, "MULTIPLE_WEBCAM_FACES")
        self.assertEqual(context.exception.details["faceCount"], 2)

    def test_missing_webcam_face_is_rejected(self) -> None:
        service = self.create_service([])
        with self.assertRaises(FaceVerificationError) as context:
            service.verify(self.card_image, self.webcam_image)
        self.assertEqual(context.exception.error_code, "WEBCAM_FACE_NOT_FOUND")

    def test_failed_quality_is_rejected(self) -> None:
        quality = make_quality(
            "webcam",
            status="fail",
            errors=("WEBCAM_FACE_TOO_BLURRY",),
        )
        service = self.create_service(
            [self.base_embedding],
            webcam_quality=quality,
        )
        with self.assertRaises(FaceVerificationError) as context:
            service.verify(self.card_image, self.webcam_image)
        self.assertEqual(context.exception.error_code, "WEBCAM_FACE_TOO_BLURRY")

    def test_result_to_dict_contains_audit_fields(self) -> None:
        result = self.create_service([self.base_embedding]).verify(
            self.card_image,
            self.webcam_image,
        ).result.to_dict()
        self.assertEqual(result["model_name"], "buffalo_l-test")
        self.assertEqual(result["portrait_rotation_degrees"], 0)
        self.assertFalse(result["liveness_checked"])
        self.assertIn("cccd_quality", result)
        self.assertIn("webcam_quality", result)

    def test_invalid_threshold_order(self) -> None:
        with self.assertRaises(ValueError):
            self.create_service(
                [self.base_embedding],
                match_threshold=0.40,
                review_threshold=0.50,
            )

    def test_empty_card_image(self) -> None:
        service = self.create_service([self.base_embedding])
        with self.assertRaises(ValueError):
            service.verify(
                np.empty((0, 0, 3), dtype=np.uint8),
                self.webcam_image,
            )

    def test_invalid_webcam_channels(self) -> None:
        service = self.create_service([self.base_embedding])
        with self.assertRaises(ValueError):
            service.verify(
                self.card_image,
                np.zeros((720, 1280), dtype=np.uint8),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
