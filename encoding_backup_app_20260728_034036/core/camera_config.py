from __future__ import annotations

from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class CameraConfig:
    """
    C?u hình k?t n?i camera b?ng OpenCV.

    Attributes:
        device_index:
            Ch? s? camera trên Windows.

        width:
            Chi?u r?ng hình ?nh mong mu?n.

        height:
            Chi?u cao hình ?nh mong mu?n.

        fps:
            S? khung hình m?i giây mong mu?n.

        backend:
            Backend OpenCV s? d?ng d? m? camera.

        fourcc:
            Chu?n mã hóa video.

        warmup_frames:
            S? frame b? qua lúc camera m?i kh?i d?ng.
    """

    device_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    backend: int = cv2.CAP_DSHOW
    fourcc: str = "MJPG"
    warmup_frames: int = 15


C922_CAMERA_CONFIG = CameraConfig(
    device_index=0,
    width=1280,
    height=720,
    fps=30,
    backend=cv2.CAP_DSHOW,
    fourcc="MJPG",
    warmup_frames=15,
)

