from __future__ import annotations

import cv2
import numpy as np

from app.core.camera_config import CameraConfig
from app.modules.face_verification.sources.base import (
    FaceImageSource,
    FaceImageSourceError,
)


class OpenCVCameraSource(FaceImageSource):
    """Nguồn ảnh webcam qua OpenCV, phù hợp Logitech C922."""

    def __init__(self, config: CameraConfig):
        self.config = config
        self._camera: cv2.VideoCapture | None = None

    @property
    def is_opened(self) -> bool:
        return self._camera is not None and self._camera.isOpened()

    def open(self) -> None:
        if self.is_opened:
            return

        camera = cv2.VideoCapture(
            self.config.device_index,
            self.config.backend,
        )
        self._camera = camera
        if not camera.isOpened():
            self.close()
            raise FaceImageSourceError(
                f"Không thể mở camera index {self.config.device_index}."
            )

        camera.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*self.config.fourcc),
        )
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        camera.set(cv2.CAP_PROP_FPS, self.config.fps)
        self._warm_up()

    def _warm_up(self) -> None:
        if self._camera is None:
            return
        for _ in range(self.config.warmup_frames):
            self._camera.read()

    def capture_frame(self) -> np.ndarray:
        if not self.is_opened:
            raise FaceImageSourceError("Camera chưa được mở.")
        assert self._camera is not None

        success, frame = self._camera.read()
        if not success or frame is None or frame.size == 0:
            raise FaceImageSourceError("Không đọc được frame từ camera.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise FaceImageSourceError("Frame camera không phải ảnh BGR 3 kênh.")
        return frame

    def get_actual_resolution(self) -> tuple[int, int]:
        if not self.is_opened:
            raise FaceImageSourceError("Camera chưa được mở.")
        assert self._camera is not None
        return (
            int(self._camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def get_actual_fps(self) -> float:
        if not self.is_opened:
            raise FaceImageSourceError("Camera chưa được mở.")
        assert self._camera is not None
        return float(self._camera.get(cv2.CAP_PROP_FPS))

    def close(self) -> None:
        if self._camera is not None:
            self._camera.release()
            self._camera = None
