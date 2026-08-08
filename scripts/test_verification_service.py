from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from app.modules.face_verification.errors import FaceVerificationError
from app.modules.face_verification.verification_service import (
    FaceVerificationService,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kiểm thử FaceVerificationService bằng hai ảnh đã lưu."
    )
    parser.add_argument("--card", required=True, help="Đường dẫn ảnh mặt trước CCCD.")
    parser.add_argument("--webcam", required=True, help="Đường dẫn ảnh webcam.")
    parser.add_argument("--match-threshold", type=float, default=0.50)
    parser.add_argument("--review-threshold", type=float, default=0.40)
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.resolve()


def read_image(image_path: Path, image_name: str):
    if not image_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {image_name}: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"OpenCV không đọc được {image_name}: {image_path}")
    return image


def main() -> None:
    args = parse_args()
    card_path = resolve_path(args.card)
    webcam_path = resolve_path(args.webcam)
    card_image = read_image(card_path, "ảnh CCCD")
    webcam_image = read_image(webcam_path, "ảnh webcam")

    output_dir = ROOT_DIR / "storage" / "debug" / "verification_service"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("FACE VERIFICATION SERVICE TEST")
    print("=" * 64)
    print(f"CCCD: {card_path}")
    print(f"Webcam: {webcam_path}")
    print(f"MATCH >= {args.match_threshold:.2f}")
    print(f"REVIEW >= {args.review_threshold:.2f}")

    service = FaceVerificationService(
        match_threshold=args.match_threshold,
        review_threshold=args.review_threshold,
    )
    try:
        output = service.verify(card_image, webcam_image)
    except FaceVerificationError as exc:
        print(f"TỪ CHỐI [{exc.error_code}]: {exc}")
        print(json.dumps(exc.details, ensure_ascii=False, indent=2))
        raise SystemExit(2) from exc

    portrait_path = output_dir / "cccd_face.jpg"
    webcam_face_path = output_dir / "webcam_face.jpg"
    json_path = output_dir / "verification_result.json"
    cv2.imwrite(str(portrait_path), output.artifacts.cccd_portrait)
    cv2.imwrite(str(webcam_face_path), output.artifacts.webcam_face)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(output.result.to_dict(), file, ensure_ascii=False, indent=2)

    print("\n" + "=" * 64)
    print("KẾT QUẢ")
    print("=" * 64)
    print(json.dumps(output.result.to_dict(), ensure_ascii=False, indent=2))
    print(f"Chân dung CCCD: {portrait_path}")
    print(f"Khuôn mặt webcam: {webcam_face_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
