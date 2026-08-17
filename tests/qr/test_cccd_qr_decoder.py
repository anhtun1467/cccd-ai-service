from __future__ import annotations

import cv2
import numpy as np

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
    assert "payload" not in result


def test_decoder_ignores_non_cccd_qr() -> None:
    result = CCCDQRDecoder().decode(
        _synthetic_card_with_qr("https://example.com/payment/123")
    )

    assert result["decoded"] is False
    assert result["structuredData"] == {}
    assert "NON_CCCD_QR_IGNORED" in result["errors"]
