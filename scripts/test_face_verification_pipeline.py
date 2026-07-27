from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from app.services.face_verification_pipeline import (
    face_verification_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ki?m th? FaceVerificationPipeline "
            "b?ng hai file ?nh."
        )
    )

    parser.add_argument(
        "--card",
        required=True,
        help="Đu?ng d?n ?nh m?t tru?c CCCD.",
    )

    parser.add_argument(
        "--webcam",
        required=True,
        help="Đu?ng d?n ?nh webcam.",
    )

    return parser.parse_args()


def resolve_file_path(
    value: str,
    file_name: str,
) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = ROOT_DIR / path

    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Không t́m th?y {file_name}: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"{file_name} không ph?i là file: {path}"
        )

    return path


def main() -> None:
    args = parse_args()

    card_path = resolve_file_path(
        args.card,
        "?nh CCCD",
    )

    webcam_path = resolve_file_path(
        args.webcam,
        "?nh webcam",
    )

    print("=" * 68)
    print("FACE VERIFICATION PIPELINE TEST")
    print("=" * 68)
    print(f"CCCD:   {card_path}")
    print(f"Webcam: {webcam_path}")
    print("=" * 68)

    card_bytes = card_path.read_bytes()
    webcam_bytes = webcam_path.read_bytes()

    print("\nĐang xác minh khuôn m?t...")

    output = face_verification_pipeline.process(
        card_image_bytes=card_bytes,
        webcam_image_bytes=webcam_bytes,
    )

    print()
    print("=" * 68)
    print("K?T QU? PIPELINE")
    print("=" * 68)

    print(
        json.dumps(
            output.to_dict(),
            ensure_ascii=False,
            indent=2,
        )
    )

    print("=" * 68)
    print(f"Request ID: {output.request_id}")
    print(f"Card debug: {output.card_image_path}")
    print(f"Webcam debug: {output.webcam_image_path}")
    print(f"Portrait debug: {output.portrait_image_path}")
    print(f"JSON debug: {output.result_json_path}")
    print("=" * 68)


if __name__ == "__main__":
    main()
