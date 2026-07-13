from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from app.modules.ocr.paddle_engine import PaddleOCREngine


def main():

    current_dir = Path(__file__).parent

    image_path = (
        current_dir.parent
        / "card_detection"
        / "output"
        / "14_final.jpg"
    )

    engine = PaddleOCREngine()

    result = engine.recognize(str(image_path))

    print("=" * 50)

    print(result.message)

    print()

    for item in result.raw_text:

        print(item)

    print("=" * 50)


if __name__ == "__main__":
    main()