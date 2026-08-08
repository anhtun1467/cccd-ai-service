from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from app.modules.face_verification.errors import FaceVerificationError
from app.modules.face_verification.portrait_extractor import (
    CCCDPortraitExtractor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trích xuất chân dung từ ảnh mặt trước CCCD."
    )
    parser.add_argument("--image", required=True, help="Đường dẫn ảnh CCCD.")
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Không mở cửa sổ preview sau khi xử lý.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = ROOT_DIR / image_path
    image_path = image_path.resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
    card_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if card_image is None or card_image.size == 0:
        raise ValueError(f"OpenCV không đọc được ảnh: {image_path}")

    output_dir = ROOT_DIR / "storage" / "debug" / "portrait_extraction"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CCCD PORTRAIT EXTRACTION")
    print("=" * 60)
    print(f"Input: {image_path}")
    print(f"Image size: {card_image.shape[1]}x{card_image.shape[0]}")
    print("Model: buffalo_l | Provider: CPUExecutionProvider")

    extractor = CCCDPortraitExtractor()
    try:
        result = extractor.extract(card_image)
    except FaceVerificationError as exc:
        print(f"TỪ CHỐI [{exc.error_code}]: {exc}")
        raise SystemExit(2) from exc

    debug_image = extractor.draw_result(card_image, result)
    portrait_path = output_dir / "portrait.jpg"
    debug_path = output_dir / "portrait_debug.jpg"
    if not cv2.imwrite(str(portrait_path), result.portrait):
        raise RuntimeError(f"Không thể lưu ảnh portrait: {portrait_path}")
    if not cv2.imwrite(str(debug_path), debug_image):
        raise RuntimeError(f"Không thể lưu ảnh debug: {debug_path}")

    print("\n" + "=" * 60)
    print("KẾT QUẢ")
    print("=" * 60)
    print(f"Method: {result.extraction_method}")
    print(f"Detection score: {result.detection_score:.4f}")
    print(f"Rotation: {result.rotation_degrees}°")
    print(f"Bounding box: {result.bbox}")
    print(f"Portrait: {portrait_path}")
    print(f"Debug: {debug_path}")

    if args.no_window:
        return
    cv2.imshow("CCCD Portrait Extraction", debug_image)
    cv2.imshow("Extracted Portrait", result.portrait)
    print("Nhấn phím bất kỳ để đóng cửa sổ.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
