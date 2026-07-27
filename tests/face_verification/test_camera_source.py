from __future__ import annotations

import cv2

from app.core.camera_config import CameraConfig
from app.modules.face_verification.sources.opencv_source import (
    OpenCVCameraSource,
)


def test_c922_capture_frame() -> None:
    """
    Ki?m tra camera có th? m? và d?c frame.

    Luu ư:
        Đây là integration test v́ c?n camera th?t.
    """
    config = CameraConfig(
        device_index=0,
        width=1280,
        height=720,
        fps=30,
        backend=cv2.CAP_DSHOW,
        fourcc="MJPG",
        warmup_frames=5,
    )

    source = OpenCVCameraSource(config)

    try:
        source.open()

        assert source.is_opened is True

        frame = source.capture_frame()

        assert frame is not None
        assert frame.size > 0
        assert frame.ndim == 3
        assert frame.shape[2] == 3

    finally:
        source.close()

    assert source.is_opened is False
