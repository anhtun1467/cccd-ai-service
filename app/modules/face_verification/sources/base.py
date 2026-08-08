from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType

import numpy as np


class FaceImageSourceError(RuntimeError):
    """Lỗi khi mở hoặc đọc ảnh từ nguồn camera."""


class FaceImageSource(ABC):
    """Giao diện chung cho webcam và các nguồn ảnh camera khác."""

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def capture_frame(self) -> np.ndarray: ...

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> "FaceImageSource":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
