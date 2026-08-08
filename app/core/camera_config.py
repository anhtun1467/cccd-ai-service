from __future__ import annotations

from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class CameraConfig:
    """Cấu hình camera OpenCV trên Windows."""

    device_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    backend: int = cv2.CAP_DSHOW
    fourcc: str = "MJPG"
    warmup_frames: int = 15

    def __post_init__(self) -> None:
        if self.device_index < 0:
            raise ValueError("device_index không được âm.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Kích thước camera phải lớn hơn 0.")
        if self.fps <= 0:
            raise ValueError("FPS phải lớn hơn 0.")
        if len(self.fourcc) != 4:
            raise ValueError("fourcc phải có đúng 4 ký tự.")
        if self.warmup_frames < 0:
            raise ValueError("warmup_frames không được âm.")


C922_CAMERA_CONFIG = CameraConfig()
