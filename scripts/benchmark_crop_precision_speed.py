from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2

from app.modules.card_detection.contour_detector import ContourDetector


def collect_images(inputs: list[str]) -> list[Path]:
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images: list[Path] = []
    for raw_path in inputs:
        path = Path(raw_path)
        if path.is_file() and path.suffix.casefold() in supported:
            images.append(path)
        elif path.is_dir():
            images.extend(
                item
                for item in path.rglob("*")
                if item.is_file() and item.suffix.casefold() in supported
            )
    return sorted(dict.fromkeys(images))


def resize_for_detection(image):
    height, width = image.shape[:2]
    if height <= 700:
        return image
    scale = 700.0 / float(height)
    return cv2.resize(
        image,
        (max(1, int(round(width * scale))), 700),
        interpolation=cv2.INTER_AREA,
    )


def benchmark_image(
    detector: ContourDetector,
    image_path: Path,
    repeat: int,
) -> dict:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return {"path": str(image_path), "error": "IMAGE_READ_FAILED"}
    resized = resize_for_detection(image)

    durations: list[float] = []
    metadata: dict = {}
    contour = None
    for _ in range(max(1, repeat)):
        started = time.perf_counter()
        contour, _, _, metadata = (
            detector.find_card_contour_candidates_from_image(resized)
        )
        durations.append((time.perf_counter() - started) * 1000.0)

    return {
        "path": str(image_path),
        "width": int(resized.shape[1]),
        "height": int(resized.shape[0]),
        "detected": contour is not None,
        "medianMs": round(statistics.median(durations), 2),
        "minimumMs": round(min(durations), 2),
        "detectionMethod": metadata.get("detectionMethod"),
        "primaryAreaRatio": metadata.get("primaryAreaRatio"),
        "primaryEdgeTouchCount": metadata.get("primaryEdgeTouchCount"),
        "wholeCardReliable": metadata.get("wholeCardReliable"),
        "houghFallbackEvaluated": metadata.get("houghFallbackEvaluated"),
        "alternateCandidateCount": len(
            metadata.get("alternateCandidates") or []
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Đo thời gian và metadata cắt CCCD mà không chạy EasyOCR."
        )
    )
    parser.add_argument("inputs", nargs="+", help="Tệp ảnh hoặc thư mục")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    image_paths = collect_images(args.inputs)
    detector = ContourDetector()
    results = [
        benchmark_image(detector, path, args.repeat)
        for path in image_paths
    ]
    successful_times = [
        float(item["medianMs"])
        for item in results
        if item.get("detected") and item.get("medianMs") is not None
    ]
    payload = {
        "imageCount": len(results),
        "detectedCount": sum(bool(item.get("detected")) for item in results),
        "medianAcrossImagesMs": (
            round(statistics.median(successful_times), 2)
            if successful_times
            else None
        ),
        "results": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
