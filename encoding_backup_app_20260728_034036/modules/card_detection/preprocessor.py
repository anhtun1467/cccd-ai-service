import cv2
import numpy as np


class ImagePreprocessor:
    """
    Tiền xử lý ảnh phục vụ phát hiện CCCD.
    """

    def __init__(self, target_height: int = 700):
        self.target_height = target_height

    def resize(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        height = image.shape[0]
        ratio = height / float(self.target_height)

        resized_width = int(image.shape[1] / ratio)
        resized = cv2.resize(image, (resized_width, self.target_height))

        return resized, ratio

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def blur(self, gray_image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(gray_image, (5, 5), 0)

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        resized, ratio = self.resize(image)
        gray = self.to_grayscale(resized)
        blurred = self.blur(gray)

        return blurred, ratio
