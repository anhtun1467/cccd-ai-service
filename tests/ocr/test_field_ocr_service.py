from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from app.modules.ocr.field_ocr_service import field_ocr_service


def choose_card_image() -> Path:
    output_dir = (
        ROOT_DIR
        / "storage"
        / "outputs"
    )

    card_images = list(
        output_dir.glob("*_card.jpg")
    )

    if not card_images:
        raise FileNotFoundError(
            "Không tìm thấy ảnh *_card.jpg"
        )

    return max(
        card_images,
        key=lambda path: path.stat().st_mtime,
    )


def main() -> None:
    card_image_path = choose_card_image()

    output_dir = (
        ROOT_DIR
        / "tests"
        / "ocr"
        / "output"
        / "field_ocr"
    )

    result = field_ocr_service.extract_fields(
        card_image_path=str(card_image_path),
        output_dir=str(output_dir),
    )

    print("=" * 80)
    print(f"Input: {card_image_path}")
    print("=" * 80)

    print(
        json.dumps(
            result["structuredData"],
            ensure_ascii=False,
            indent=2,
        )
    )

    print("=" * 80)

    for field_name, field_result in result[
        "fieldResults"
    ].items():
        print(
            f"{field_name:<20} "
            f"value={field_result['value']} "
            f"confidence={field_result['averageConfidence']}"
        )

    print("=" * 80)
    print(
        "Debug:",
        result["debug"],
    )


if __name__ == "__main__":
    main()
