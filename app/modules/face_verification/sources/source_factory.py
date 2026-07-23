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
    Factory khởi tạo nguồn ảnh khuôn mặt.
    """

    @staticmethod
    def create_c922(
        config: CameraConfig | None = None,
    ) -> FaceImageSource:
        """
        Khởi tạo nguồn Logitech C922.

        Args:
            config:
                Cấu hình tùy chỉnh. Nếu không truyền vào,
                hệ thống sử dụng C922_CAMERA_CONFIG.
        """
        return OpenCVCameraSource(
            config=config or C922_CAMERA_CONFIG
        )
