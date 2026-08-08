from __future__ import annotations

import cv2
import numpy as np

from app.modules.face_verification.models import FaceEmbeddingResult
from app.modules.face_verification.quality import FaceQualityEvaluator


def make_face() -> FaceEmbeddingResult:
    return FaceEmbeddingResult(
        embedding=np.ones(512, dtype=np.float32),
        bbox=np.asarray([120, 80, 360, 400], dtype=np.float32),
        detection_score=0.98,
        landmarks=np.asarray(
            [[190, 180], [290, 180], [240, 240], [205, 310], [275, 310]],
            dtype=np.float32,
        ),
        pose=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
    )


def test_clear_centered_webcam_face_passes() -> None:
    image = np.zeros((480, 480, 3), dtype=np.uint8)
    tile = 16
    for y in range(80, 400, tile):
        for x in range(120, 360, tile):
            value = 70 if ((x // tile) + (y // tile)) % 2 else 190
            image[y:y + tile, x:x + tile] = value

    quality = FaceQualityEvaluator().evaluate(image, make_face(), "webcam")

    assert quality.is_acceptable is True
    assert quality.status == "pass"
    assert quality.sharpness > 45
    assert quality.errors == ()


def test_severely_blurred_webcam_face_is_rejected() -> None:
    image = np.full((480, 480, 3), 128, dtype=np.uint8)
    quality = FaceQualityEvaluator().evaluate(image, make_face(), "webcam")

    assert quality.is_acceptable is False
    assert "WEBCAM_FACE_TOO_BLURRY" in quality.errors


def test_eye_landmarks_are_used_for_roll_when_pose_is_missing() -> None:
    image = np.zeros((480, 480, 3), dtype=np.uint8)
    noise = np.random.default_rng(123).integers(
        40,
        210,
        size=(320, 240, 3),
        dtype=np.uint8,
    )
    image[80:400, 120:360] = cv2.GaussianBlur(noise, (3, 3), 0)
    face = make_face()
    tilted_face = FaceEmbeddingResult(
        embedding=face.embedding,
        bbox=face.bbox,
        detection_score=face.detection_score,
        landmarks=np.asarray(
            [[190, 160], [290, 230], [240, 250], [205, 310], [275, 330]],
            dtype=np.float32,
        ),
        pose=None,
    )

    quality = FaceQualityEvaluator().evaluate(
        image,
        tilted_face,
        "webcam",
    )

    assert quality.roll is not None
    assert abs(quality.roll) > 30
    assert "WEBCAM_FACE_ROLL_INVALID" in quality.errors

