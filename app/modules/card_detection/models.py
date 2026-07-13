from dataclasses import dataclass

import numpy as np


@dataclass
class CardCorners:
    """
    Lưu 4 góc của CCCD theo thứ tự:
    top_left, top_right, bottom_right, bottom_left.
    """

    top_left: np.ndarray
    top_right: np.ndarray
    bottom_right: np.ndarray
    bottom_left: np.ndarray

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.top_left,
                self.top_right,
                self.bottom_right,
                self.bottom_left,
            ],
            dtype="float32",
        )


@dataclass
class CardDetectionResult:
    """
    Kết quả phát hiện CCCD.
    """

    success: bool
    message: str
    card_image: np.ndarray | None = None
    corners: CardCorners | None = None