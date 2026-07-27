from __future__ import annotations

import cv2
import numpy as np

from app.core.camera_config import CameraConfig
from app.modules.face_verification.sources.base import (
    FaceImageSource,
    FaceImageSourceError,
)


class OpenCVCameraSource(FaceImageSource):
    """
    Ngu?n ?nh s? d?ng OpenCV.

    Class này có th? dùng cho Logitech C922 và các webcam
    tuong thích chu?n camera c?a Windows.
    """

    def __init__(self, config: CameraConfig):
        self.config = config
        self._camera: cv2.VideoCapture | None = None

    @property
    def is_opened(self) -> bool:
        """
        Ki?m tra camera hi?n có dang m? hay không.
        """
        return (
            self._camera is not None
            and self._camera.isOpened()
        )

    def open(self) -> None:
        """
        M? camera và thi?t l?p d? phân gi?i.
        """
        if self.is_opened:
            return

        self._camera = cv2.VideoCapture(
            self.config.device_index,
            self.config.backend,
        )

        if not self._camera.isOpened():
            self.close()

            raise FaceImageSourceError(
                "Không th? m? camera v?i index "
                f"{self.config.device_index}."
            )

        fourcc_code = cv2.VideoWriter_fourcc(
            *self.config.fourcc
        )

        self._camera.set(
            cv2.CAP_PROP_FOURCC,
            fourcc_code,
        )

        self._camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self.config.width,
        )

        self._camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self.config.height,
        )

        self._camera.set(
            cv2.CAP_PROP_FPS,
            self.config.fps,
        )

        self._warm_up()

    def _warm_up(self) -> None:
        """
        Đ?c b? m?t s? frame d?u.

        Vi?c này giúp camera có th?i gian:
        - t? cân b?ng sáng;
        - t? l?y nét;
        - ?n d?nh h́nh ?nh.
        """
        if self._camera is None:
            return

        for _ in range(self.config.warmup_frames):
            self._camera.read()

    def capture_frame(self) -> np.ndarray:
        """
        Ch?p và tr? v? m?t frame BGR.
        """
        if not self.is_opened:
            raise FaceImageSourceError(
                "Camera chua du?c m?."
            )

        assert self._camera is not None

        success, frame = self._camera.read()

        if not success or frame is None:
            raise FaceImageSourceError(
                "Không d?c du?c frame t? camera."
            )

        if frame.size == 0:
            raise FaceImageSourceError(
                "Frame camera tr? v? b? r?ng."
            )

        return frame

    def get_actual_resolution(self) -> tuple[int, int]:
        """
        Tr? v? d? phân gi?i camera dang s? d?ng.

        Returns:
            Tuple (width, height).
        """
        if not self.is_opened:
            raise FaceImageSourceError(
                "Camera chua du?c m?."
            )

        assert self._camera is not None

        width = int(
            self._camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            self._camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        return width, height

    def get_actual_fps(self) -> float:
        """
        Tr? v? FPS camera dang du?c c?u h́nh.
        """
        if not self.is_opened:
            raise FaceImageSourceError(
                "Camera chua du?c m?."
            )

        assert self._camera is not None

        return float(
            self._camera.get(cv2.CAP_PROP_FPS)
        )

    def close(self) -> None:
        """
        Gi?i phóng camera.
        """
        if self._camera is not None:
            self._camera.release()
            self._camera = None
