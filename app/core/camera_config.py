from __future__ import annotations

from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class CameraConfig:
    """
    Cấu hình kết nối camera bằng OpenCV.

    Attributes:
        device_index:
            Chỉ số camera trên Windows.

        width:
            Chiều rộng hình ảnh mong muốn.

        height:
            Chiều cao hình ảnh mong muốn.

        fps:
            Số khung hình mỗi giây mong muốn.

        backend:
            Backend OpenCV sử dụng để mở camera.

        fourcc:
            Chuẩn mã hóa video.

        warmup_frames:
            Số frame bỏ qua lúc camera mới khởi động.
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
