from __future__ import annotations

import os

import cv2
import pytest

from app.core.camera_config import CameraConfig
from app.modules.face_verification.sources.opencv_source import (
    OpenCVCameraSource,
)


def test_c922_default_configuration() -> None:
    config = CameraConfig()
    assert config.device_index == 0
    assert config.width == 1280
    assert config.height == 720
    assert config.fps == 30
    assert config.backend == cv2.CAP_DSHOW
    assert config.fourcc == "MJPG"


@pytest.mark.skipif(
    os.getenv("RUN_CAMERA_TEST") != "1",
    reason="Đặt RUN_CAMERA_TEST=1 để chạy kiểm thử với webcam thật.",
)
def test_c922_capture_frame_integration() -> None:
    config = CameraConfig(warmup_frames=5)
    source = OpenCVCameraSource(config)

    try:
        source.open()
        frame = source.capture_frame()
        assert source.is_opened is True
        assert frame is not None
        assert frame.size > 0
        assert frame.ndim == 3
        assert frame.shape[2] == 3
    finally:
        source.close()

    assert source.is_opened is False
