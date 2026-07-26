from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from app.modules.face_verification.portrait_extractor import (
    PortraitExtractionResult,
)
from app.modules.face_verification.verification_service import (
    FaceVerificationService,
)


class FakePortraitExtractor:
    """
    Portrait extractor giả để unit test không cần tải InsightFace.
    """

    def __init__(self) -> None:
        self.portrait = np.full(
            shape=(200, 150, 3),
            fill_value=128,
            dtype=np.uint8,
        )

    def extract(
        self,
        card_image: np.ndarray,
    ) -> PortraitExtractionResult:
        return PortraitExtractionResult(
            portrait=self.portrait.copy(),
            bbox=(10, 20, 160, 220),
            detection_score=0.91,
            extraction_method="fake_portrait_extractor",
            source_width=card_image.shape[1],
            source_height=card_image.shape[0],
        )


class FakeEmbedder:
    """
    Embedder giả trả lần lượt embedding CCCD và webcam.
    """

    def __init__(
        self,
        cccd_embedding: np.ndarray,
        webcam_embedding: np.ndarray,
    ) -> None:
        self.results = [
            SimpleNamespace(
                embedding=cccd_embedding,
                detection_score=0.92,
            ),
            SimpleNamespace(
                embedding=webcam_embedding,
                detection_score=0.88,
            ),
        ]

    def extract_single(
        self,
        image: np.ndarray,
    ):
        if not self.results:
            raise RuntimeError(
                "FakeEmbedder không còn kết quả giả."
            )

        return self.results.pop(0)


class FaceVerificationServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.card_image = np.zeros(
            shape=(600, 1000, 3),
            dtype=np.uint8,
        )

        self.webcam_image = np.zeros(
            shape=(720, 1280, 3),
            dtype=np.uint8,
        )

        self.base_embedding = np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        )

    def create_service(
        self,
        webcam_embedding: np.ndarray,
        match_threshold: float = 0.50,
        review_threshold: float = 0.40,
    ) -> FaceVerificationService:
        return FaceVerificationService(
            portrait_extractor=FakePortraitExtractor(),
            embedder=FakeEmbedder(
                cccd_embedding=self.base_embedding,
                webcam_embedding=webcam_embedding,
            ),
            match_threshold=match_threshold,
            review_threshold=review_threshold,
        )

    def test_match_status(self) -> None:
        """
        Hai vector giống nhau phải trả về MATCH.
        """

        service = self.create_service(
            webcam_embedding=np.array(
                [1.0, 0.0, 0.0],
                dtype=np.float32,
            )
        )

        output = service.verify(
            card_image=self.card_image,
            webcam_image=self.webcam_image,
        )

        self.assertEqual(
            output.result.status,
            "match",
        )

        self.assertTrue(
            output.result.is_match
        )

        self.assertFalse(
            output.result.needs_review
        )

        self.assertAlmostEqual(
            output.result.similarity,
            1.0,
            places=5,
        )

    def test_review_status(self) -> None:
        """
        Similarity nằm giữa 0.40 và 0.50 phải trả REVIEW.
        """

        similarity = 0.45

        webcam_embedding = np.array(
            [
                similarity,
                np.sqrt(1.0 - similarity**2),
                0.0,
            ],
            dtype=np.float32,
        )

        service = self.create_service(
            webcam_embedding=webcam_embedding
        )

        output = service.verify(
            card_image=self.card_image,
            webcam_image=self.webcam_image,
        )

        self.assertEqual(
            output.result.status,
            "review",
        )

        self.assertFalse(
            output.result.is_match
        )

        self.assertTrue(
            output.result.needs_review
        )

        self.assertAlmostEqual(
            output.result.similarity,
            0.45,
            places=4,
        )

    def test_not_match_status(self) -> None:
        """
        Hai vector vuông góc có similarity bằng 0,
        phải trả NOT_MATCH.
        """

        service = self.create_service(
            webcam_embedding=np.array(
                [0.0, 1.0, 0.0],
                dtype=np.float32,
            )
        )

        output = service.verify(
            card_image=self.card_image,
            webcam_image=self.webcam_image,
        )

        self.assertEqual(
            output.result.status,
            "not_match",
        )

        self.assertFalse(
            output.result.is_match
        )

        self.assertFalse(
            output.result.needs_review
        )

        self.assertAlmostEqual(
            output.result.similarity,
            0.0,
            places=5,
        )

    def test_result_to_dict(self) -> None:
        """
        Kiểm tra dữ liệu đầu ra phù hợp để trả về API.
        """

        service = self.create_service(
            webcam_embedding=np.array(
                [1.0, 0.0, 0.0],
                dtype=np.float32,
            )
        )

        output = service.verify(
            card_image=self.card_image,
            webcam_image=self.webcam_image,
        )

        result_dict = output.result.to_dict()

        self.assertEqual(
            result_dict["status"],
            "match",
        )

        self.assertTrue(
            result_dict["is_match"]
        )

        self.assertIn(
            "similarity",
            result_dict,
        )

        self.assertIn(
            "processing_time_ms",
            result_dict,
        )

        self.assertEqual(
            result_dict["portrait_method"],
            "fake_portrait_extractor",
        )

        self.assertEqual(
            tuple(result_dict["portrait_bbox"]),
            (10, 20, 160, 220),
        )

    def test_invalid_threshold_order(self) -> None:
        """
        review_threshold phải nhỏ hơn match_threshold.
        """

        with self.assertRaises(ValueError):
            FaceVerificationService(
                portrait_extractor=FakePortraitExtractor(),
                embedder=FakeEmbedder(
                    cccd_embedding=self.base_embedding,
                    webcam_embedding=self.base_embedding,
                ),
                match_threshold=0.40,
                review_threshold=0.50,
            )

    def test_empty_card_image(self) -> None:
        service = self.create_service(
            webcam_embedding=self.base_embedding
        )

        empty_image = np.empty(
            shape=(0, 0, 3),
            dtype=np.uint8,
        )

        with self.assertRaises(ValueError):
            service.verify(
                card_image=empty_image,
                webcam_image=self.webcam_image,
            )

    def test_invalid_webcam_channels(self) -> None:
        service = self.create_service(
            webcam_embedding=self.base_embedding
        )

        grayscale_image = np.zeros(
            shape=(720, 1280),
            dtype=np.uint8,
        )

        with self.assertRaises(ValueError):
            service.verify(
                card_image=self.card_image,
                webcam_image=grayscale_image,
            )


if __name__ == "__main__":
    unittest.main(
        verbosity=2
    )
