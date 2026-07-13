from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from app.modules.ocr.text_normalizer import TextNormalizer


def main():
    raw = [
        "Vict Nana",
        "Ha No",
        "24/0311995",
        "Hova ten / Full name:",
        "CONC DAN",
    ]

    normalizer = TextNormalizer()

    result = normalizer.normalize(raw)

    print("=" * 50)

    for line in result:
        print(line)

    print("=" * 50)


if __name__ == "__main__":
    main()