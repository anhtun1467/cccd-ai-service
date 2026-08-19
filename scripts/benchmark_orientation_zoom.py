from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

import cv2

from app.modules.card_detection.detector import CardDetector
from app.modules.card_detection.fast_orientation import FastCardOrientation
from app.modules.qr.cccd_qr_decoder import CCCDQRDecoder


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def benchmark(image_path: Path, loops: int, skip_qr: bool) -> dict:
    source = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if source is None or source.size == 0:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")

    rotations = {
        "0": source,
        "90": cv2.rotate(source, cv2.ROTATE_90_CLOCKWISE),
        "180": cv2.rotate(source, cv2.ROTATE_180),
        "270": cv2.rotate(source, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }
    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="cccd_orientation_") as temp_dir:
        for rotation_name, rotated_source in rotations.items():
            temporary_path = Path(temp_dir) / f"rotation_{rotation_name}.jpg"
            if not cv2.imwrite(str(temporary_path), rotated_source):
                raise OSError(f"Không thể tạo ảnh benchmark {rotation_name}")

            detection_started = time.perf_counter()
            detection = CardDetector().detect_from_path(str(temporary_path))
            detection_ms = (time.perf_counter() - detection_started) * 1000.0
            card_image = detection["cardImage"]

            orientation_times: list[float] = []
            orientation = {}
            classifier = FastCardOrientation()
            for _ in range(max(1, loops)):
                orientation_started = time.perf_counter()
                orientation = classifier.analyze(card_image)
                orientation_times.append(
                    (time.perf_counter() - orientation_started) * 1000.0
                )

            qr_summary = None
            if not skip_qr:
                qr_result = CCCDQRDecoder().decode(card_image)
                qr_summary = {
                    "decoded": bool(qr_result.get("decoded")),
                    "status": qr_result.get("status"),
                    "attemptCount": int(qr_result.get("attemptCount", 0) or 0),
                    "selectedVariant": qr_result.get("selectedVariant"),
                    "regionDetected": bool(qr_result.get("regionDetected")),
                    "elapsedMs": float(qr_result.get("elapsedMs", 0.0) or 0.0),
                }

            geometry = detection.get("geometry", {})
            rows.append({
                "inputRotationDegrees": int(rotation_name),
                "inputShape": list(rotated_source.shape[:2]),
                "cardShape": list(card_image.shape[:2]),
                "geometryRotationDegrees": geometry.get(
                    "geometryRotationDegrees"
                ),
                "outputScaleLimited": bool(geometry.get("outputScaleLimited")),
                "detectorMs": round(detection_ms, 2),
                "orientation": {
                    "reliable": bool(orientation.get("reliable")),
                    "rotationDegrees": int(
                        orientation.get("rotationDegrees", 0) or 0
                    ),
                    "source": orientation.get("source"),
                    "confidence": float(
                        orientation.get("confidence", 0.0) or 0.0
                    ),
                    "p50Ms": round(statistics.median(orientation_times), 3),
                    "p95Ms": round(percentile(orientation_times, 0.95), 3),
                },
                "qr": qr_summary,
            })

    return {
        "image": str(image_path),
        "loops": max(1, loops),
        # Kết quả cố ý không chứa payload QR hoặc dữ liệu CCCD.
        "privacy": "NO_QR_PAYLOAD_OR_CCCD_FIELDS",
        "rotations": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Đo detector/nhận chiều/QR mà không in dữ liệu CCCD",
    )
    parser.add_argument("image", type=Path, help="Ảnh CCCD dùng để đo")
    parser.add_argument("--loops", type=int, default=20)
    parser.add_argument("--skip-qr", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(
        benchmark(arguments.image, arguments.loops, arguments.skip_qr),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
