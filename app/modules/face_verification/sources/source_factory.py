from __future__ import annotations

from app.core.camera_config import C922_CAMERA_CONFIG, CameraConfig
from app.modules.face_verification.sources.base import FaceImageSource
from app.modules.face_verification.sources.opencv_source import (
    OpenCVCameraSource,
)


class FaceImageSourceFactory:
    @staticmethod
    def create_c922(
        config: CameraConfig | None = None,
    ) -> FaceImageSource:
        return OpenCVCameraSource(config or C922_CAMERA_CONFIG)
