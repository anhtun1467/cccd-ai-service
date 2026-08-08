from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from app.services.face_verification_pipeline import FaceVerificationPipeline


def encode_image(
    size: tuple[int, int] = (320, 240),
    color: tuple[int, int, int] = (10, 20, 30),
) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_decoder_returns_bgr_image() -> None:
    pipeline = FaceVerificationPipeline(
        provider=object(),  # type: ignore[arg-type]
        save_debug=False,
        max_image_pixels=100_000,
    )
    image = pipeline._decode_image(encode_image(), "ảnh thử")

    assert image.shape == (240, 320, 3)
    # JPEG có sai số nén nhỏ; kiểm tra thứ tự BGR thay vì giá trị tuyệt đối.
    blue, green, red = [int(value) for value in image[0, 0]]
    assert blue > green > red


def test_decoder_rejects_invalid_bytes() -> None:
    pipeline = FaceVerificationPipeline(
        provider=object(),  # type: ignore[arg-type]
        save_debug=False,
    )
    with pytest.raises(ValueError, match="Không thể giải mã"):
        pipeline._decode_image(b"not-an-image", "ảnh lỗi")


def test_decoder_rejects_excessive_pixel_count() -> None:
    pipeline = FaceVerificationPipeline(
        provider=object(),  # type: ignore[arg-type]
        save_debug=False,
        max_image_pixels=10_000,
    )
    with pytest.raises(ValueError, match="vượt quá giới hạn"):
        pipeline._decode_image(encode_image(), "ảnh lớn")

