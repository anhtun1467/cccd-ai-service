from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from app.modules.card_detection.image_loader import ImageLoader
from app.modules.card_detection.preprocessor import ImagePreprocessor
from app.modules.card_detection.contour_detector import ContourDetector


def main() -> None:
    current_dir = Path(__file__).parent
    image_path = current_dir / "sample_cccd.jpg"
    output_dir = current_dir / "output"
    output_dir.mkdir(exist_ok=True)

    loader = ImageLoader()
    preprocessor = ImagePreprocessor()
    contour_detector = ContourDetector()

    image = loader.load(str(image_path))
    resized, ratio = preprocessor.resize(image)

    card_contour, mask, contours = contour_detector.find_card_contour_from_image(
        resized
    )

    all_contours_debug = resized.copy()
    card_debug = resized.copy()

    cv2.drawContours(all_contours_debug, contours, -1, (0, 255, 255), 2)

    if card_contour is not None:
        cv2.drawContours(card_debug, [card_contour], -1, (0, 255, 0), 5)

        x, y, w, h = cv2.boundingRect(card_contour)
        area = cv2.contourArea(card_contour)
        aspect_ratio = w / float(h)

        print("Tìm thấy contour CCCD")
        print("Số điểm:", len(card_contour))
        print("Area:", area)
        print("Bounding box:", x, y, w, h)
        print("Aspect ratio:", aspect_ratio)
    else:
        print("Chưa tìm thấy contour CCCD")

    cv2.imwrite(str(output_dir / "06_card_mask.jpg"), mask)
    cv2.imwrite(str(output_dir / "07_all_contours.jpg"), all_contours_debug)
    cv2.imwrite(str(output_dir / "08_card_contour.jpg"), card_debug)

    print("Contour Detector OK")
    print("Số contour kiểm tra:", len(contours))
    print("Resize ratio:", ratio)


if __name__ == "__main__":
    main()