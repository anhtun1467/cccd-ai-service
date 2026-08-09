from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from app.modules.face_verification.errors import FaceVerificationError
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


class FakeIntegratedService:
    def __init__(self, *, reject_prepared: bool = False) -> None:
        self.reject_prepared = reject_prepared
        self.prepared_calls = 0
        self.card_calls = 0
        self.output = SimpleNamespace(result=SimpleNamespace(status="match"))

    def verify_prepared_portrait(self, *args, **kwargs):
        self.prepared_calls += 1
        if self.reject_prepared:
            raise FaceVerificationError(
                "CCCD_FACE_NOT_FOUND",
                "Không thấy mặt trong crop OCR.",
            )
        return self.output

    def verify(self, *args, **kwargs):
        self.card_calls += 1
        return self.output


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_image())


def test_process_from_ocr_paths_prefers_prepared_portrait(
    tmp_path: Path,
) -> None:
    card_path = tmp_path / "card.jpg"
    portrait_path = tmp_path / "portrait.jpg"
    write_image(card_path)
    write_image(portrait_path)
    service = FakeIntegratedService()
    pipeline = FaceVerificationPipeline(
        provider=SimpleNamespace(service=service),
        save_debug=False,
    )

    output = pipeline.process_from_ocr_paths(
        card_image_path=card_path,
        portrait_image_path=portrait_path,
        webcam_image_bytes=encode_image(),
    )

    assert output.reference_source == "ocr_portrait_crop"
    assert service.prepared_calls == 1
    assert service.card_calls == 0


def test_process_from_ocr_paths_falls_back_to_flattened_card(
    tmp_path: Path,
) -> None:
    card_path = tmp_path / "card.jpg"
    portrait_path = tmp_path / "portrait.jpg"
    write_image(card_path)
    write_image(portrait_path)
    service = FakeIntegratedService(reject_prepared=True)
    pipeline = FaceVerificationPipeline(
        provider=SimpleNamespace(service=service),
        save_debug=False,
    )

    output = pipeline.process_from_ocr_paths(
        card_image_path=card_path,
        portrait_image_path=portrait_path,
        webcam_image_bytes=encode_image(),
    )

    assert output.reference_source == "ocr_card_image_fallback"
    assert service.prepared_calls == 1
    assert service.card_calls == 1


def test_process_from_ocr_paths_rejects_corrupted_server_reference(
    tmp_path: Path,
) -> None:
    card_path = tmp_path / "card.jpg"
    card_path.write_bytes(b"not-an-image")
    pipeline = FaceVerificationPipeline(
        provider=SimpleNamespace(service=FakeIntegratedService()),
        save_debug=False,
    )

    with pytest.raises(FaceVerificationError) as context:
        pipeline.process_from_ocr_paths(
            card_image_path=card_path,
            webcam_image_bytes=encode_image(),
        )

    assert context.value.error_code == "OCR_REFERENCE_IMAGE_INVALID"
    assert context.value.status_code == 410
