from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np

from app.modules.card_detection.adaptive_quality_enhancer import (
    AdaptiveQualityEnhancer,
)
from app.modules.card_detection.detector import CardDetector


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Đo bộ tăng cường thích ứng trên các ảnh CCCD bị giảm sáng "
            "và làm mờ có kiểm soát. Đây là benchmark chất lượng ảnh, "
            "không thay thế benchmark độ chính xác OCR có ground truth."
        )
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("datasets/source/images"),
    )
    parser.add_argument(
        "--brightness-factor",
        type=float,
        default=0.48,
    )
    parser.add_argument(
        "--blur-sigma",
        type=float,
        default=1.10,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Tệp JSON tùy chọn để lưu báo cáo chi tiết.",
    )
    return parser.parse_args()


def image_metrics(image: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {
        "brightness": round(float(np.mean(gray)), 3),
        "percentile99": round(float(np.percentile(gray, 99)), 3),
        "laplacianVariance": round(
            float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            3,
        ),
    }


def degrade_image(
    image: np.ndarray,
    brightness_factor: float,
    blur_sigma: float,
) -> np.ndarray:
    dimmed = np.clip(
        image.astype(np.float32) * float(brightness_factor) + 3.0,
        0,
        255,
    ).astype(np.uint8)
    if blur_sigma <= 0:
        return dimmed
    return cv2.GaussianBlur(
        dimmed,
        (5, 5),
        sigmaX=float(blur_sigma),
        sigmaY=float(blur_sigma),
    )


def build_report(arguments: argparse.Namespace) -> dict[str, Any]:
    detector = CardDetector()
    enhancer = AdaptiveQualityEnhancer()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    paths = sorted(
        path
        for path in arguments.image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    for path in paths:
        try:
            card = detector.detect_from_path(str(path))["cardImage"]
            degraded = degrade_image(
                card,
                brightness_factor=arguments.brightness_factor,
                blur_sigma=arguments.blur_sigma,
            )
            profile, variants = enhancer.build_ocr_variants(degraded)
            selected = next(
                (
                    candidate["image"]
                    for candidate in variants
                    if candidate["name"] == "low_light_deblur"
                ),
                None,
            )
            if selected is None and variants:
                selected = variants[0]["image"]
            if selected is None:
                raise RuntimeError("Không tạo được biến thể thích ứng")

            rows.append({
                "image": str(path),
                "shapePreserved": selected.shape == degraded.shape,
                "profile": profile,
                "before": image_metrics(degraded),
                "after": image_metrics(selected),
            })
        except Exception as error:
            failures.append({
                "image": str(path),
                "error": str(error),
            })

    before_brightness = [
        row["before"]["brightness"] for row in rows
    ]
    after_brightness = [
        row["after"]["brightness"] for row in rows
    ]
    before_laplacian = [
        row["before"]["laplacianVariance"] for row in rows
    ]
    after_laplacian = [
        row["after"]["laplacianVariance"] for row in rows
    ]

    return {
        "benchmarkType": "IMAGE_QUALITY_ONLY",
        "accuracyClaimAllowed": False,
        "degradation": {
            "brightnessFactor": arguments.brightness_factor,
            "blurSigma": arguments.blur_sigma,
        },
        "summary": {
            "processed": len(rows),
            "failed": len(failures),
            "shapePreserved": all(
                bool(row["shapePreserved"]) for row in rows
            ),
            "allCandidatesBrighter": all(
                after > before + 20.0
                for before, after in zip(
                    before_brightness,
                    after_brightness,
                )
            ),
            "allCandidatesSharper": all(
                after > before
                for before, after in zip(
                    before_laplacian,
                    after_laplacian,
                )
            ),
            "medianBrightnessBefore": (
                round(median(before_brightness), 3)
                if before_brightness
                else None
            ),
            "medianBrightnessAfter": (
                round(median(after_brightness), 3)
                if after_brightness
                else None
            ),
            "medianLaplacianBefore": (
                round(median(before_laplacian), 3)
                if before_laplacian
                else None
            ),
            "medianLaplacianAfter": (
                round(median(after_laplacian), 3)
                if after_laplacian
                else None
            ),
        },
        "rows": rows,
        "failures": failures,
    }


def main() -> None:
    arguments = parse_arguments()
    report = build_report(arguments)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
        print(json.dumps({
            "output": str(arguments.output),
            "summary": report["summary"],
        }, ensure_ascii=False, indent=2))
    else:
        print(payload)


if __name__ == "__main__":
    main()
