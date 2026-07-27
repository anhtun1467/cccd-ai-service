from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Type

import numpy as np


class FaceImageSourceError(RuntimeError):
    """
    L?i x?y ra khi m? ho?c d?c ?nh t? ngu?n camera.
    """


class FaceImageSource(ABC):
    """
    Interface chung cho các ngu?n ?nh khuôn m?t.

    Sau này Logitech C922, camera IP ho?c ?nh t? máy ch?m công
    d?u có th? tri?n khai theo interface này.
    """

    @abstractmethod
    def open(self) -> None:
        """
        M? k?t n?i t?i ngu?n ?nh.
        """

    @abstractmethod
    def capture_frame(self) -> np.ndarray:
        """
        Ch?p m?t frame.

        Returns:
            ?nh d?ng NumPy BGR.

        Raises:
            FaceImageSourceError:
                N?u ngu?n chua m? ho?c không d?c du?c frame.
        """

    @abstractmethod
    def close(self) -> None:
        """
        Ðóng k?t n?i t?i ngu?n ?nh.
        """

    def __enter__(self) -> "FaceImageSource":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

