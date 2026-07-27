from __future__ import annotations

from abc import ABC, abstractmethod

from app.modules.ocr.models import OCRResult


class BaseOCREngine(ABC):
    """
    Interface cho mọi OCR Engine.
    """

    @abstractmethod
    def recognize(
        self,
        image_path: str,
    ) -> OCRResult:
        """
        Đọc văn bản từ ảnh.

        Parameters
        ----------
        image_path : str
            Đường dẫn ảnh.

        Returns
        -------
        OCRResult
        """
        pass
