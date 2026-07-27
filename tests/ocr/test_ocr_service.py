from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from app.modules.ocr.service import ocr_service


def main() -> None:
    current_dir = Path(__file__).parent

    image_path = (
        current_dir.parent
        / "card_detection"
        / "output"
        / "14_final.jpg"
    )

    result = ocr_service.extract_cccd_info(str(image_path))

    print("=" * 60)
    print("OCR + Parser + Validator Result")
    print("=" * 60)

    print(
        json.dumps(
            {
                "structuredData": result["structuredData"],
                "validation": result["validation"],
            },
            ensure_ascii=False,
            indent=4,
        )
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
