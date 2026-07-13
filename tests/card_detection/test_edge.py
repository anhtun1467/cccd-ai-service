from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from app.modules.card_detection.image_loader import ImageLoader
from app.modules.card_detection.preprocessor import ImagePreprocessor
from app.modules.card_detection.edge_detector import EdgeDetector


def main() -> None:
    current_dir = Path(__file__).parent
    image_path = current_dir / "sample_cccd.jpg"
    output_dir = current_dir / "output"
    output_dir.mkdir(exist_ok=True)

    loader = ImageLoader()
    preprocessor = ImagePreprocessor()
    edge_detector = EdgeDetector()

    image = loader.load(str(image_path))
    blurred, ratio = preprocessor.preprocess(image)
    edge = edge_detector.detect(blurred)

    cv2.imwrite(str(output_dir / "05_edge.jpg"), edge)

    print("Edge Detector OK")
    print("Edge shape:", edge.shape)
    print("Resize ratio:", ratio)


if __name__ == "__main__":
    main()