from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from app.modules.ocr.easyocr_engine import EasyOCREngine


def main():

    current_dir = Path(__file__).parent

    image_path = (
        current_dir.parent
        / "card_detection"
        / "output"
        / "14_final.jpg"
    )

    engine = EasyOCREngine()

    result = engine.recognize(str(image_path))

    print("=" * 60)

    print(result.message)

    print()

    for text in result.raw_text:

        print(text)

    print("=" * 60)


if __name__ == "__main__":
    main()