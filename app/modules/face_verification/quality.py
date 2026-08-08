from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, sqrt

import cv2
import numpy as np

from app.modules.face_verification.models import (
    FaceEmbeddingResult,
    FaceImageKind,
    FaceQualityResult,
)


@dataclass(frozen=True)
class FaceQualityPolicy:
    min_face_size: int
    hard_min_sharpness: float
    warning_min_sharpness: float
    hard_min_brightness: float
    warning_min_brightness: float
    hard_max_brightness: float
    warning_max_brightness: float
    hard_min_face_height_ratio: float
    warning_min_face_height_ratio: float
    hard_max_face_height_ratio: float
    warning_max_face_height_ratio: float
    hard_max_yaw: float
    warning_max_yaw: float
    hard_max_pitch: float
    warning_max_pitch: float
    hard_max_roll: float
    warning_max_roll: float
    warning_max_center_offset: float


CCCD_QUALITY_POLICY = FaceQualityPolicy(
    min_face_size=32,
    hard_min_sharpness=2.0,
    warning_min_sharpness=8.0,
    hard_min_brightness=12.0,
    warning_min_brightness=28.0,
    hard_max_brightness=248.0,
    warning_max_brightness=235.0,
    hard_min_face_height_ratio=0.04,
    warning_min_face_height_ratio=0.07,
    hard_max_face_height_ratio=0.98,
    warning_max_face_height_ratio=0.90,
    hard_max_yaw=45.0,
    warning_max_yaw=32.0,
    hard_max_pitch=40.0,
    warning_max_pitch=30.0,
    hard_max_roll=40.0,
    warning_max_roll=28.0,
    warning_max_center_offset=1.0,
)


WEBCAM_QUALITY_POLICY = FaceQualityPolicy(
    min_face_size=80,
    hard_min_sharpness=12.0,
    warning_min_sharpness=45.0,
    hard_min_brightness=25.0,
    warning_min_brightness=50.0,
    hard_max_brightness=238.0,
    warning_max_brightness=215.0,
    hard_min_face_height_ratio=0.14,
    warning_min_face_height_ratio=0.24,
    hard_max_face_height_ratio=0.92,
    warning_max_face_height_ratio=0.78,
    hard_max_yaw=35.0,
    warning_max_yaw=22.0,
    hard_max_pitch=32.0,
    warning_max_pitch=20.0,
    hard_max_roll=30.0,
    warning_max_roll=17.0,
    warning_max_center_offset=0.42,
)


class FaceQualityEvaluator:
    """Đánh giá nét, sáng, kích thước, vị trí và góc khuôn mặt."""

    def __init__(
        self,
        cccd_policy: FaceQualityPolicy = CCCD_QUALITY_POLICY,
        webcam_policy: FaceQualityPolicy = WEBCAM_QUALITY_POLICY,
    ) -> None:
        self.cccd_policy = cccd_policy
        self.webcam_policy = webcam_policy

    def evaluate(
        self,
        image: np.ndarray,
        face: FaceEmbeddingResult,
        source: FaceImageKind,
    ) -> FaceQualityResult:
        self._validate_image(image)

        policy = (
            self.cccd_policy
            if source == "cccd"
            else self.webcam_policy
        )

        image_height, image_width = image.shape[:2]
        x1 = max(0, min(face.x1, image_width - 1))
        y1 = max(0, min(face.y1, image_height - 1))
        x2 = max(x1 + 1, min(face.x2, image_width))
        y2 = max(y1 + 1, min(face.y2, image_height))

        face_crop = image[y1:y2, x1:x2]
        if face_crop.size == 0:
            raise ValueError("Vùng khuôn mặt bị rỗng.")

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())

        face_width = x2 - x1
        face_height = y2 - y1
        face_height_ratio = face_height / max(1, image_height)
        center_offset = self._center_offset(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            image_width=image_width,
            image_height=image_height,
        )

        pitch, yaw, roll = self._extract_pose(face)
        errors: list[str] = []
        warnings: list[str] = []

        if min(face_width, face_height) < policy.min_face_size:
            errors.append(f"{source.upper()}_FACE_TOO_SMALL")

        self._append_range_issue(
            value=sharpness,
            hard_min=policy.hard_min_sharpness,
            warning_min=policy.warning_min_sharpness,
            hard_code=f"{source.upper()}_FACE_TOO_BLURRY",
            warning_code=f"{source.upper()}_FACE_SLIGHTLY_BLURRY",
            errors=errors,
            warnings=warnings,
        )

        self._append_range_issue(
            value=brightness,
            hard_min=policy.hard_min_brightness,
            warning_min=policy.warning_min_brightness,
            hard_code=f"{source.upper()}_FACE_TOO_DARK",
            warning_code=f"{source.upper()}_FACE_DARK",
            errors=errors,
            warnings=warnings,
        )

        self._append_upper_issue(
            value=brightness,
            hard_max=policy.hard_max_brightness,
            warning_max=policy.warning_max_brightness,
            hard_code=f"{source.upper()}_FACE_TOO_BRIGHT",
            warning_code=f"{source.upper()}_FACE_BRIGHT",
            errors=errors,
            warnings=warnings,
        )

        self._append_range_issue(
            value=face_height_ratio,
            hard_min=policy.hard_min_face_height_ratio,
            warning_min=policy.warning_min_face_height_ratio,
            hard_code=f"{source.upper()}_FACE_TOO_FAR",
            warning_code=f"{source.upper()}_FACE_FAR",
            errors=errors,
            warnings=warnings,
        )

        self._append_upper_issue(
            value=face_height_ratio,
            hard_max=policy.hard_max_face_height_ratio,
            warning_max=policy.warning_max_face_height_ratio,
            hard_code=f"{source.upper()}_FACE_TOO_CLOSE",
            warning_code=f"{source.upper()}_FACE_CLOSE",
            errors=errors,
            warnings=warnings,
        )

        self._append_pose_issue(
            value=yaw,
            hard_max=policy.hard_max_yaw,
            warning_max=policy.warning_max_yaw,
            source=source,
            axis="YAW",
            errors=errors,
            warnings=warnings,
        )
        self._append_pose_issue(
            value=pitch,
            hard_max=policy.hard_max_pitch,
            warning_max=policy.warning_max_pitch,
            source=source,
            axis="PITCH",
            errors=errors,
            warnings=warnings,
        )
        self._append_pose_issue(
            value=roll,
            hard_max=policy.hard_max_roll,
            warning_max=policy.warning_max_roll,
            source=source,
            axis="ROLL",
            errors=errors,
            warnings=warnings,
        )

        if center_offset > policy.warning_max_center_offset:
            warnings.append(f"{source.upper()}_FACE_OFF_CENTER")

        if errors:
            status = "fail"
        elif warnings:
            status = "warning"
        else:
            status = "pass"

        return FaceQualityResult(
            source=source,
            status=status,
            is_acceptable=not errors,
            sharpness=sharpness,
            brightness=brightness,
            face_width=face_width,
            face_height=face_height,
            face_height_ratio=face_height_ratio,
            center_offset=center_offset,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    @staticmethod
    def _center_offset(
        *,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        image_width: int,
        image_height: int,
    ) -> float:
        face_center_x = (x1 + x2) / 2.0
        face_center_y = (y1 + y2) / 2.0
        normalized_x = abs(face_center_x - image_width / 2.0) / max(
            1.0,
            image_width / 2.0,
        )
        normalized_y = abs(face_center_y - image_height / 2.0) / max(
            1.0,
            image_height / 2.0,
        )
        return min(1.0, sqrt(normalized_x**2 + normalized_y**2) / sqrt(2.0))

    @staticmethod
    def _extract_pose(
        face: FaceEmbeddingResult,
    ) -> tuple[float | None, float | None, float | None]:
        if face.pose is not None:
            pose = np.asarray(face.pose, dtype=np.float32).reshape(-1)
            if pose.size >= 3 and np.all(np.isfinite(pose[:3])):
                # InsightFace trả về thứ tự pitch, yaw, roll.
                return float(pose[0]), float(pose[1]), float(pose[2])

        if face.landmarks is not None:
            landmarks = np.asarray(face.landmarks, dtype=np.float32)
            if landmarks.shape[0] >= 2:
                left_eye = landmarks[0]
                right_eye = landmarks[1]
                roll = degrees(
                    atan2(
                        float(right_eye[1] - left_eye[1]),
                        float(right_eye[0] - left_eye[0]),
                    )
                )
                return None, None, roll

        return None, None, None

    @staticmethod
    def _append_range_issue(
        *,
        value: float,
        hard_min: float,
        warning_min: float,
        hard_code: str,
        warning_code: str,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        if value < hard_min:
            errors.append(hard_code)
        elif value < warning_min:
            warnings.append(warning_code)

    @staticmethod
    def _append_upper_issue(
        *,
        value: float,
        hard_max: float,
        warning_max: float,
        hard_code: str,
        warning_code: str,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        if value > hard_max:
            errors.append(hard_code)
        elif value > warning_max:
            warnings.append(warning_code)

    @staticmethod
    def _append_pose_issue(
        *,
        value: float | None,
        hard_max: float,
        warning_max: float,
        source: FaceImageKind,
        axis: str,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        if value is None:
            return
        absolute_value = abs(value)
        if absolute_value > hard_max:
            errors.append(f"{source.upper()}_FACE_{axis}_INVALID")
        elif absolute_value > warning_max:
            warnings.append(f"{source.upper()}_FACE_{axis}_WARNING")

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if image is None or not isinstance(image, np.ndarray):
            raise TypeError("Ảnh đánh giá chất lượng phải là numpy.ndarray.")
        if image.size == 0:
            raise ValueError("Ảnh đánh giá chất lượng bị rỗng.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Ảnh đánh giá chất lượng phải là ảnh BGR 3 kênh.")

