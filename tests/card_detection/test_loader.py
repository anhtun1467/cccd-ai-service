from pathlib import Path
import sys

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from app.modules.card_detection.image_loader import ImageLoader


def main() -> None:
    current_dir = Path(__file__).parent

    image_path = current_dir / "sample_cccd.jpg"
    output_path = current_dir / "output" / "loader.jpg"

    loader = ImageLoader()
    image = loader.load(str(image_path))

    print("Đọc ảnh thành công")
    print("Kích thước ảnh:", image.shape)

    cv2.imwrite(str(output_path), image)

    print("Đã lưu ảnh test tại:", output_path)


if __name__ == "__main__":
    main()