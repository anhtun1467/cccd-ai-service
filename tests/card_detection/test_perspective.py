from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from app.modules.card_detection.image_loader import ImageLoader
from app.modules.card_detection.preprocessor import ImagePreprocessor
from app.modules.card_detection.contour_detector import ContourDetector
from app.modules.card_detection.perspective_transformer import PerspectiveTransformer


def main() -> None:
    current_dir = Path(__file__).parent
    image_path = current_dir / "sample_cccd.jpg"
    output_dir = current_dir / "output"
    output_dir.mkdir(exist_ok=True)

    loader = ImageLoader()
    preprocessor = ImagePreprocessor()
    contour_detector = ContourDetector()
    transformer = PerspectiveTransformer()

    image = loader.load(str(image_path))

    resized, ratio = preprocessor.resize(image)

    card_contour, mask, contours = contour_detector.find_card_contour_from_image(
        resized
    )

    if card_contour is None:
        print("Không tìm thấy contour CCCD")
        return

    warped = transformer.transform(resized, card_contour)

    cv2.imwrite(str(output_dir / "09_warped_card.jpg"), warped)

    print("Perspective Transform OK")
    print("Warped shape:", warped.shape)
    print("Resize ratio:", ratio)


if __name__ == "__main__":
    main()