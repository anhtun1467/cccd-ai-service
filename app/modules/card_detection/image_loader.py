from pathlib import Path

import cv2
import numpy as np

from app.core.exceptions import BadRequestException


class ImageLoader:
    """
    Chịu trách nhiệm đọc ảnh từ ổ đĩa.

    Sau này có thể mở rộng:
        - Camera
        - Webcam
        - URL
        - Base64
    """

    def load(self, image_path: str) -> np.ndarray:

        path = Path(image_path)

        if not path.exists():
            raise BadRequestException(
                f"Không tìm thấy ảnh: {image_path}"
            )

        image = cv2.imread(str(path))

        if image is None:
            raise BadRequestException(
                "Không đọc được ảnh."
            )

        return image