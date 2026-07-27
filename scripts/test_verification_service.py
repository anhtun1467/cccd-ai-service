from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from app.modules.face_verification.verification_service import (
    FaceVerificationService,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ki?m th? FaceVerificationService "
            "b?ng ?nh CCCD và ?nh webcam dă luu."
        )
    )

    parser.add_argument(
        "--card",
        required=True,
        type=str,
        help="Đu?ng d?n ?nh m?t tru?c CCCD.",
    )

    parser.add_argument(
        "--webcam",
        required=True,
        type=str,
        help="Đu?ng d?n ?nh webcam.",
    )

    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.50,
        help="Ngu?ng MATCH.",
    )

    parser.add_argument(
        "--review-threshold",
        type=float,
        default=0.40,
        help="Ngu?ng REVIEW.",
    )

    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = ROOT_DIR / path

    return path.resolve()


def read_image(
    image_path: Path,
    image_name: str,
):
    if not image_path.exists():
        raise FileNotFoundError(
            f"Không t́m th?y {image_name}: {image_path}"
        )

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"OpenCV không d?c du?c {image_name}: "
            f"{image_path}"
        )

    return image


def main() -> None:
    args = parse_args()

    card_path = resolve_path(args.card)
    webcam_path = resolve_path(args.webcam)

    card_image = read_image(
        card_path,
        "?nh CCCD",
    )

    webcam_image = read_image(
        webcam_path,
        "?nh webcam",
    )

    output_dir = (
        ROOT_DIR
        / "storage"
        / "debug"
        / "verification_service"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 64)
    print("FACE VERIFICATION SERVICE TEST")
    print("=" * 64)
    print(f"CCCD: {card_path}")
    print(f"Webcam: {webcam_path}")
    print(
        f"Match threshold: "
        f"{args.match_threshold:.4f}"
    )
    print(
        f"Review threshold: "
        f"{args.review_threshold:.4f}"
    )
    print("=" * 64)

    print("\nĐang kh?i t?o service...")

    service = FaceVerificationService(
        match_threshold=args.match_threshold,
        review_threshold=args.review_threshold,
    )

    print("Đang xác minh khuôn m?t...")

    output = service.verify(
        card_image=card_image,
        webcam_image=webcam_image,
    )

    result = output.result

    portrait_path = (
        output_dir
        / "cccd_portrait.jpg"
    )

    cv2.imwrite(
        str(portrait_path),
        output.artifacts.cccd_portrait,
    )

    json_path = (
        output_dir
        / "verification_result.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result.to_dict(),
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 64)
    print("K?T QU?")
    print("=" * 64)
    print(
        json.dumps(
            result.to_dict(),
            ensure_ascii=False,
            indent=2,
        )
    )
    print("=" * 64)
    print(f"Portrait: {portrait_path}")
    print(f"JSON result: {json_path}")


if __name__ == "__main__":
    main()
