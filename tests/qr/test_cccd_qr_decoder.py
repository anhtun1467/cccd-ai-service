from __future__ import annotations

import cv2
import numpy as np

import app.modules.qr.cccd_qr_decoder as decoder_module
from app.modules.qr.cccd_qr_decoder import CCCDQRDecoder


def _synthetic_card_with_qr(payload: str) -> np.ndarray:
    qr = cv2.QRCodeEncoder_create().encode(payload)
    qr = cv2.resize(
        qr,
        None,
        fx=7,
        fy=7,
        interpolation=cv2.INTER_NEAREST,
    )
    qr = cv2.copyMakeBorder(
        qr,
        30,
        30,
        30,
        30,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    qr_bgr = cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR)
    card = np.full((630, 1000, 3), 238, dtype=np.uint8)
    height, width = qr_bgr.shape[:2]
    card[20 : 20 + height, 980 - width : 980] = qr_bgr
    return card


def test_decoder_reads_synthetic_cccd_qr_without_exposing_payload() -> None:
    payload = (
        "042096015766||ĐINH XUÂN HOÀNG|24111996|Nam|"
        "Lâm Trung Thủy, Đức Thọ, Hà Tĩnh|01122021"
    )

    result = CCCDQRDecoder().decode(_synthetic_card_with_qr(payload))

    assert result["decoded"] is True
    assert result["structuredData"]["idNumber"] == "042096015766"
    assert result["structuredData"]["fullName"] == "ĐINH XUÂN HOÀNG"
    assert result["attemptCount"] == 1
    assert result["status"] == "DECODED_VALID"
    assert result["regionDetected"] is True
    assert len(result["polygon"]) == 4
    assert result["boundingBox"]["width"] > 0
    assert result["boundingBox"]["height"] > 0
    assert result["debugCrop"].size > 0
    assert "payload" not in result


def test_decoder_ignores_non_cccd_qr() -> None:
    result = CCCDQRDecoder().decode(
        _synthetic_card_with_qr("https://example.com/payment/123")
    )

    assert result["decoded"] is False
    assert result["structuredData"] == {}
    assert result["status"] == "DECODED_NON_CCCD"
    assert result["regionDetected"] is True
    assert "NON_CCCD_QR_IGNORED" in result["errors"]
    assert any(
        item["code"] == "NON_CCCD_QR_IGNORED"
        for item in result["errorDetails"]
    )


def test_decoder_reads_upside_down_card_and_maps_bottom_left_region() -> None:
    payload = (
        "042096015766||ĐINH XUÂN HOÀNG|24111996|Nam|"
        "Lâm Trung Thủy, Đức Thọ, Hà Tĩnh|01122021"
    )
    rotated = cv2.rotate(
        _synthetic_card_with_qr(payload),
        cv2.ROTATE_180,
    )

    result = CCCDQRDecoder().decode(rotated)

    assert result["decoded"] is True
    box = result["boundingBox"]
    center_x = box["x"] + box["width"] / 2
    center_y = box["y"] + box["height"] / 2
    assert center_x < rotated.shape[1] * 0.50
    assert center_y > rotated.shape[0] * 0.50
    assert "bottomLeft" in result["searchRegions"]


def test_dense_bottom_left_variant_maps_polygon_back_to_card(
    monkeypatch,
) -> None:
    payload = (
        "042096015766||ĐINH XUÂN HOÀNG|24111996|Nam|"
        "Lâm Trung Thủy, Đức Thọ, Hà Tĩnh|01122021"
    )

    class _Point:
        def __init__(self, x: int, y: int) -> None:
            self.x = x
            self.y = y

    class _Position:
        top_left = _Point(100, 100)
        top_right = _Point(200, 100)
        bottom_right = _Point(200, 200)
        bottom_left = _Point(100, 200)

    class _Barcode:
        valid = True
        text = payload
        position = _Position()

    calls = 0

    def fake_read_barcodes(*args, **kwargs):
        nonlocal calls
        calls += 1
        return [_Barcode()] if calls == 4 else []

    monkeypatch.setattr(
        decoder_module.zxingcpp,
        "read_barcodes",
        fake_read_barcodes,
    )
    card = np.full((630, 1000, 3), 230, dtype=np.uint8)

    result = CCCDQRDecoder(time_budget_ms=500).decode(card)

    assert result["decoded"] is True
    assert result["selectedVariant"] == "bottom_left_dense_detail"
    assert result["attemptCount"] == 4
    box = result["boundingBox"]
    assert box["x"] + box["width"] / 2 < card.shape[1] * 0.50
    assert box["y"] + box["height"] / 2 > card.shape[0] * 0.50
