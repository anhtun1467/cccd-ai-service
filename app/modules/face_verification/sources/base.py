from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Type

import numpy as np


class FaceImageSourceError(RuntimeError):
    """
    Lỗi xảy ra khi mở hoặc đọc ảnh từ nguồn camera.
    """


class FaceImageSource(ABC):
    """
    Interface chung cho các nguồn ảnh khuôn mặt.

    Sau này Logitech C922, camera IP hoặc ảnh từ máy chấm công
    đều có thể triển khai theo interface này.
    """

    @abstractmethod
    def open(self) -> None:
        """
        Mở kết nối tới nguồn ảnh.
        """

    @abstractmethod
    def capture_frame(self) -> np.ndarray:
        """
        Chụp một frame.

        Returns:
            Ảnh dạng NumPy BGR.

        Raises:
            FaceImageSourceError:
                Nếu nguồn chưa mở hoặc không đọc được frame.
        """

    @abstractmethod
    def close(self) -> None:
        """
        Đóng kết nối tới nguồn ảnh.
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
