from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi import HTTPException

import app.api.ocr as ocr_module
from app.utils.image_validator import check_image_quality


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _checkerboard(
    height: int = 600,
    width: int = 900,
    block_size: int = 20,
    low: int = 20,
    high: int = 220,
) -> np.ndarray:
    rows, columns = np.indices((height, width))
    mask = ((rows // block_size) + (columns // block_size)) % 2
    gray = np.where(mask == 0, low, high).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_clear_image_is_accepted() -> None:
    result = check_image_quality(_checkerboard())

    assert result["is_valid"] is True
    assert result["blur_score"] >= 100.0
    assert result["brightness_score"] >= 60.0
    assert result["reason"] == "Hợp lệ"


def test_blurred_image_is_rejected() -> None:
    blurred = cv2.GaussianBlur(_checkerboard(), (51, 51), 15)

    result = check_image_quality(blurred)

    assert result["is_valid"] is False
    assert result["blur_score"] < 100.0
    assert "mờ" in result["reason"].lower()


def test_dark_but_sharp_image_is_rejected() -> None:
    dark_image = _checkerboard(low=0, high=40)

    result = check_image_quality(dark_image)

    assert result["is_valid"] is False
    assert result["brightness_score"] < 60.0
    assert "thiếu sáng" in result["reason"].lower()


def test_empty_image_is_rejected() -> None:
    result = check_image_quality(np.array([], dtype=np.uint8))

    assert result["is_valid"] is False
    assert result["blur_score"] == 0.0
    assert result["brightness_score"] == 0.0


def test_unsupported_channel_count_is_rejected() -> None:
    image = np.zeros((100, 100, 2), dtype=np.uint8)

    result = check_image_quality(image)

    assert result["is_valid"] is False
    assert "không được hỗ trợ" in result["reason"].lower()


@pytest.mark.anyio
async def test_ocr_endpoint_rejects_low_quality_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "dark.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((300, 500, 3), dtype=np.uint8))

    async def fake_save_upload_file(_file: object) -> Path:
        return image_path

    pipeline_called = False

    def fake_process_cccd_image(_path: Path) -> dict[str, object]:
        nonlocal pipeline_called
        pipeline_called = True
        return {}

    monkeypatch.setattr(ocr_module, "save_upload_file", fake_save_upload_file)
    monkeypatch.setattr(
        ocr_module.ocr_pipeline_service,
        "process_cccd_image",
        fake_process_cccd_image,
    )

    with pytest.raises(HTTPException) as error:
        await ocr_module.ocr_cccd(file=None)  # type: ignore[arg-type]

    assert error.value.status_code == 400
    assert pipeline_called is False


@pytest.mark.anyio
async def test_ocr_endpoint_accepts_valid_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "clear.jpg"
    assert cv2.imwrite(str(image_path), _checkerboard())

    async def fake_save_upload_file(_file: object) -> Path:
        return image_path

    expected_result = {"idNumber": "001234567890"}

    def fake_process_cccd_image(path: Path) -> dict[str, str]:
        assert path == image_path
        return expected_result

    monkeypatch.setattr(ocr_module, "save_upload_file", fake_save_upload_file)
    monkeypatch.setattr(
        ocr_module.ocr_pipeline_service,
        "process_cccd_image",
        fake_process_cccd_image,
    )

    response = await ocr_module.ocr_cccd(file=None)  # type: ignore[arg-type]

    assert response.success is True
    assert response.data == expected_result

