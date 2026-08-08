from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from app.modules.face_verification.errors import FaceVerificationError
from app.services.face_verification_pipeline import face_verification_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kiểm thử FaceVerificationPipeline bằng hai file ảnh."
    )
    parser.add_argument("--card", required=True, help="Đường dẫn ảnh mặt trước CCCD.")
    parser.add_argument("--webcam", required=True, help="Đường dẫn ảnh webcam.")
    return parser.parse_args()


def resolve_file_path(value: str, file_name: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT_DIR / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy {file_name}: {path}")
    if not path.is_file():
        raise ValueError(f"{file_name} không phải là file: {path}")
    return path


def main() -> None:
    args = parse_args()
    card_path = resolve_file_path(args.card, "ảnh CCCD")
    webcam_path = resolve_file_path(args.webcam, "ảnh webcam")

    print("=" * 68)
    print("FACE VERIFICATION PIPELINE TEST")
    print("=" * 68)
    print(f"CCCD: {card_path}")
    print(f"Webcam: {webcam_path}")

    try:
        output = face_verification_pipeline.process(
            card_image_bytes=card_path.read_bytes(),
            webcam_image_bytes=webcam_path.read_bytes(),
        )
    except FaceVerificationError as exc:
        print(f"TỪ CHỐI [{exc.error_code}]: {exc}")
        print(json.dumps(exc.details, ensure_ascii=False, indent=2))
        raise SystemExit(2) from exc

    print("\n" + "=" * 68)
    print("KẾT QUẢ PIPELINE")
    print("=" * 68)
    print(json.dumps(output.to_dict(), ensure_ascii=False, indent=2))
    print(f"Request ID: {output.request_id}")
    print(f"Card debug: {output.card_image_path}")
    print(f"Webcam debug: {output.webcam_image_path}")
    print(f"CCCD face debug: {output.portrait_image_path}")
    print(f"Webcam face debug: {output.webcam_face_path}")
    print(f"JSON debug: {output.result_json_path}")


if __name__ == "__main__":
    main()
