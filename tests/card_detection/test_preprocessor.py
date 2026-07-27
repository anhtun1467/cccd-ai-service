from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from app.modules.card_detection.image_loader import ImageLoader
from app.modules.card_detection.preprocessor import ImagePreprocessor


def main() -> None:
    current_dir = Path(__file__).parent
    image_path = current_dir / "sample_cccd.jpg"
    output_dir = current_dir / "output"
    output_dir.mkdir(exist_ok=True)

    loader = ImageLoader()
    preprocessor = ImagePreprocessor()

    image = loader.load(str(image_path))
    resized, ratio = preprocessor.resize(image)
    gray = preprocessor.to_grayscale(resized)
    blurred = preprocessor.blur(gray)

    cv2.imwrite(str(output_dir / "01_original.jpg"), image)
    cv2.imwrite(str(output_dir / "02_resized.jpg"), resized)
    cv2.imwrite(str(output_dir / "03_grayscale.jpg"), gray)
    cv2.imwrite(str(output_dir / "04_blur.jpg"), blurred)

    print("Preprocessor OK")
    print("Original shape:", image.shape)
    print("Resized shape:", resized.shape)
    print("Resize ratio:", ratio)


if __name__ == "__main__":
    main()
