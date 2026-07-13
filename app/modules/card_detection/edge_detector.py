import cv2
import numpy as np


class EdgeDetector:
    """
    Phát hiện cạnh trong ảnh bằng Canny Edge Detection.
    """

    def __init__(self, threshold1: int = 50, threshold2: int = 150):
        self.threshold1 = threshold1
        self.threshold2 = threshold2

    def detect(self, image: np.ndarray) -> np.ndarray:
        return cv2.Canny(image, self.threshold1, self.threshold2)
    