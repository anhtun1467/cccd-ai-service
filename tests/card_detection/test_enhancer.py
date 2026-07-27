from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from app.modules.card_detection.image_loader import ImageLoader
from app.modules.card_detection.preprocessor import ImagePreprocessor
from app.modules.card_detection.contour_detector import ContourDetector
from app.modules.card_detection.perspective_transformer import PerspectiveTransformer
from app.modules.card_detection.enhancer import ImageEnhancer


def main() -> None:
    current_dir = Path(__file__).parent
    image_path = current_dir / "sample_cccd.jpg"
    output_dir = current_dir / "output"
    output_dir.mkdir(exist_ok=True)

    loader = ImageLoader()
    preprocessor = ImagePreprocessor()
    contour_detector = ContourDetector()
    transformer = PerspectiveTransformer()
    enhancer = ImageEnhancer()

    image = loader.load(str(image_path))

    resized, ratio = preprocessor.resize(image)

    card_contour, mask, contours = contour_detector.find_card_contour_from_image(
        resized
    )

    if card_contour is None:
        print("Không tìm thấy contour CCCD")
        return

    warped = transformer.transform(resized, card_contour)

    enhanced_images = enhancer.enhance(warped)

    cv2.imwrite(str(output_dir / "09_warped_card.jpg"), warped)
    cv2.imwrite(str(output_dir / "10_brightness.jpg"), enhanced_images["brightness"])
    cv2.imwrite(str(output_dir / "11_clahe.jpg"), enhanced_images["clahe"])
    cv2.imwrite(str(output_dir / "12_sharpen.jpg"), enhanced_images["sharpen"])
    cv2.imwrite(str(output_dir / "13_denoise.jpg"), enhanced_images["denoise"])
    cv2.imwrite(str(output_dir / "14_final.jpg"), enhanced_images["final"])

    print("Image Enhancer OK")
    print("Warped shape:", warped.shape)
    print("Final shape:", enhanced_images["final"].shape)
    print("Resize ratio:", ratio)
    print("Đã lưu ảnh debug vào:", output_dir)


if __name__ == "__main__":
    main()
