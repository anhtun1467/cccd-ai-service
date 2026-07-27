from __future__ import annotations

from app.core.camera_config import (
    C922_CAMERA_CONFIG,
    CameraConfig,
)
from app.modules.face_verification.sources.base import (
    FaceImageSource,
)
from app.modules.face_verification.sources.opencv_source import (
    OpenCVCameraSource,
)


class FaceImageSourceFactory:
    """
    Factory kh?i t?o ngu?n ?nh khuôn m?t.
    """

    @staticmethod
    def create_c922(
        config: CameraConfig | None = None,
    ) -> FaceImageSource:
        """
        Kh?i t?o ngu?n Logitech C922.

        Args:
            config:
                C?u h́nh tùy ch?nh. N?u không truy?n vào,
                h? th?ng s? d?ng C922_CAMERA_CONFIG.
        """
        return OpenCVCameraSource(
            config=config or C922_CAMERA_CONFIG
        )
