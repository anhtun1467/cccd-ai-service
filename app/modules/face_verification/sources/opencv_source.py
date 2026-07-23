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
    Nguồn ảnh sử dụng OpenCV.

    Class này có thể dùng cho Logitech C922 và các webcam
    tương thích chuẩn camera của Windows.
    """

    def __init__(self, config: CameraConfig):
        self.config = config
        self._camera: cv2.VideoCapture | None = None

    @property
    def is_opened(self) -> bool:
        """
        Kiểm tra camera hiện có đang mở hay không.
        """
        return (
            self._camera is not None
            and self._camera.isOpened()
        )

    def open(self) -> None:
        """
        Mở camera và thiết lập độ phân giải.
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
                "Không thể mở camera với index "
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
        Đọc bỏ một số frame đầu.

        Việc này giúp camera có thời gian:
        - tự cân bằng sáng;
        - tự lấy nét;
        - ổn định hình ảnh.
        """
        if self._camera is None:
            return

        for _ in range(self.config.warmup_frames):
            self._camera.read()

    def capture_frame(self) -> np.ndarray:
        """
        Chụp và trả về một frame BGR.
        """
        if not self.is_opened:
            raise FaceImageSourceError(
                "Camera chưa được mở."
            )

        assert self._camera is not None

        success, frame = self._camera.read()

        if not success or frame is None:
            raise FaceImageSourceError(
                "Không đọc được frame từ camera."
            )

        if frame.size == 0:
            raise FaceImageSourceError(
                "Frame camera trả về bị rỗng."
            )

        return frame

    def get_actual_resolution(self) -> tuple[int, int]:
        """
        Trả về độ phân giải camera đang sử dụng.

        Returns:
            Tuple (width, height).
        """
        if not self.is_opened:
            raise FaceImageSourceError(
                "Camera chưa được mở."
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
        Trả về FPS camera đang được cấu hình.
        """
        if not self.is_opened:
            raise FaceImageSourceError(
                "Camera chưa được mở."
            )

        assert self._camera is not None

        return float(
            self._camera.get(cv2.CAP_PROP_FPS)
        )

    def close(self) -> None:
        """
        Giải phóng camera.
        """
        if self._camera is not None:
            self._camera.release()
            self._camera = None
