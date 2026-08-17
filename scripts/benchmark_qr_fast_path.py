from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from app.core.config import settings  # noqa: E402
from app.modules.card_detection.detector import CardDetector  # noqa: E402
from app.modules.qr.cccd_qr_decoder import CCCDQRDecoder  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(inputs: list[str]) -> list[Path]:
    images: list[Path] = []
    for input_value in inputs:
        path = Path(input_value).expanduser().resolve()
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES:
            images.append(path)
            continue
        if path.is_dir():
            images.extend(
                candidate
                for candidate in sorted(path.rglob("*"))
                if candidate.is_file()
                and candidate.suffix.casefold() in IMAGE_SUFFIXES
            )
    return list(dict.fromkeys(images))


def benchmark_image(
    image_path: Path,
    detector: CardDetector,
    decoder: CCCDQRDecoder,
    repeat: int,
    debug_directory: str,
) -> dict[str, Any]:
    detection_times: list[float] = []
    qr_times: list[float] = []
    last_qr_result: dict[str, Any] = {}
    error: str | None = None

    for _ in range(repeat):
        try:
            detection_started = time.perf_counter()
            detection = detector.detect_from_path(
                image_path=str(image_path),
                output_dir=debug_directory,
            )
            detection_times.append(
                (time.perf_counter() - detection_started) * 1000.0
            )
            last_qr_result = decoder.decode(detection.get("cardImage"))
            qr_times.append(float(last_qr_result.get("elapsedMs", 0.0)))
        except Exception as caught_error:
            error = str(caught_error)
            break

    return {
        "image": str(image_path),
        "success": error is None,
        "error": error,
        "repeat": repeat,
        "cardDetectionMedianMs": (
            round(statistics.median(detection_times), 2)
            if detection_times
            else None
        ),
        "qrDecodeMedianMs": (
            round(statistics.median(qr_times), 2) if qr_times else None
        ),
        "qrDecoded": bool(last_qr_result.get("decoded")),
        "qrAttemptCount": int(
            last_qr_result.get("attemptCount", 0) or 0
        ),
        "qrFormat": last_qr_result.get("format"),
        "qrProvidedFields": list(
            last_qr_result.get("providedFields", []) or []
        ),
        "qrErrors": list(last_qr_result.get("errors", []) or []),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Card Detection + QR Fast Path mà không nạp EasyOCR. "
            "Kết quả không in payload hoặc dữ liệu cá nhân trong QR."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Một hoặc nhiều tệp/thư mục ảnh CCCD.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Số lần đo mỗi ảnh (mặc định: 3).",
    )
    parser.add_argument(
        "--qr-budget-ms",
        type=float,
        default=float(settings.qr_decode_budget_ms),
        help="Ngân sách mềm cho QR decoder.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repeat = max(1, int(args.repeat))
    images = collect_images(args.inputs)
    if not images:
        print(json.dumps({"error": "Không tìm thấy ảnh hợp lệ"}, ensure_ascii=False))
        return 2

    detector = CardDetector()
    decoder = CCCDQRDecoder(time_budget_ms=float(args.qr_budget_ms))
    with tempfile.TemporaryDirectory(prefix="cccd_qr_benchmark_") as debug_dir:
        results = [
            benchmark_image(
                image_path=image_path,
                detector=detector,
                decoder=decoder,
                repeat=repeat,
                debug_directory=debug_dir,
            )
            for image_path in images
        ]

    successful = [item for item in results if item["success"]]
    decoded = [item for item in successful if item["qrDecoded"]]
    qr_times = [
        float(item["qrDecodeMedianMs"])
        for item in successful
        if item["qrDecodeMedianMs"] is not None
    ]
    report = {
        "summary": {
            "imageCount": len(results),
            "successfulCount": len(successful),
            "qrDecodedCount": len(decoded),
            "qrDecodeRate": round(
                len(decoded) / len(successful),
                4,
            )
            if successful
            else 0.0,
            "qrDecodeMedianAcrossImagesMs": round(
                statistics.median(qr_times),
                2,
            )
            if qr_times
            else None,
            "rawPayloadLogged": False,
        },
        "results": results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if len(successful) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
