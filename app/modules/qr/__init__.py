"""Giải mã và hợp nhất dữ liệu QR trên thẻ CCCD/Căn cước."""

from app.modules.qr.cccd_qr_decoder import CCCDQRDecoder
from app.modules.qr.cccd_qr_parser import CCCDQRParser

__all__ = ["CCCDQRDecoder", "CCCDQRParser"]
