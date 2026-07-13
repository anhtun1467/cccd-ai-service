from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from app.modules.card_detection.detector import CardDetector


def main() -> None:
    current_dir = Path(__file__).parent
    image_path = current_dir / "sample_cccd.jpg"
    output_dir = current_dir / "output"

    detector = CardDetector()

    result = detector.detect_from_path(
        image_path=str(image_path),
        output_dir=str(output_dir),
    )

    print(result["message"])
    print("Resize ratio:", result["resizeRatio"])
    print("Card shape:", result["cardImage"].shape)
    print("Enhanced shape:", result["enhancedImage"].shape)


if __name__ == "__main__":
    main()