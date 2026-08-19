from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _safe_polygon(value: Any) -> np.ndarray | None:
    try:
        points = np.asarray(value, dtype=np.float32).reshape(-1, 2)
    except (TypeError, ValueError):
        return None
    if len(points) < 4 or not np.all(np.isfinite(points)):
        return None
    return points[:4]


def draw_qr_overlay(
    image: np.ndarray,
    qr_result: dict[str, Any],
    source_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Vẽ vùng QR hoặc vùng đã tìm mà không hiển thị payload PII."""
    overlay = image.copy()
    output_height, output_width = overlay.shape[:2]
    if source_size is None:
        source_width, source_height = output_width, output_height
    else:
        source_width = max(int(source_size[0]), 1)
        source_height = max(int(source_size[1]), 1)
    scale_x = output_width / float(source_width)
    scale_y = output_height / float(source_height)

    polygon = _safe_polygon(qr_result.get("polygon"))
    if polygon is not None:
        polygon[:, 0] *= scale_x
        polygon[:, 1] *= scale_y
        integer_polygon = np.rint(polygon).astype(np.int32)
        color = (40, 190, 40) if qr_result.get("decoded") else (0, 165, 255)
        status = "QR DECODED" if qr_result.get("decoded") else "QR DETECTED"
        cv2.polylines(
            overlay,
            [integer_polygon],
            True,
            color,
            max(2, int(round(min(output_width, output_height) * 0.004))),
            cv2.LINE_AA,
        )
        label_x = int(np.min(integer_polygon[:, 0]))
        label_y = max(24, int(np.min(integer_polygon[:, 1])) - 8)
        cv2.putText(
            overlay,
            status,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
        return overlay

    search_regions = qr_result.get("searchRegions")
    regions: list[tuple[str, dict[str, Any]]] = []
    if isinstance(search_regions, dict):
        regions = [
            (str(name), region)
            for name, region in search_regions.items()
            if isinstance(region, dict)
        ]
    if not regions:
        search_region = qr_result.get("searchRegion")
        if isinstance(search_region, dict):
            regions = [("topRight", search_region)]

    for index, (region_name, search_region) in enumerate(regions):
        x1 = int(round(float(search_region.get("x", 0)) * scale_x))
        y1 = int(round(float(search_region.get("y", 0)) * scale_y))
        x2 = int(round(
            (float(search_region.get("x", 0))
             + float(search_region.get("width", 0)))
            * scale_x
        ))
        y2 = int(round(
            (float(search_region.get("y", 0))
             + float(search_region.get("height", 0)))
            * scale_y
        ))
        color = (0, 215, 255)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = (
            "QR SEARCH - NOT DETECTED"
            if len(regions) == 1
            else f"QR SEARCH {index + 1}: {region_name}"
        )
        cv2.putText(
            overlay,
            label,
            (max(0, x1), max(24, y1 + 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return overlay


def save_qr_debug_images(
    card_image: np.ndarray,
    qr_result: dict[str, Any],
    debug_dir: str | Path,
) -> dict[str, str]:
    qr_dir = Path(debug_dir) / "qr"
    qr_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    overlay = draw_qr_overlay(card_image, qr_result)
    overlay_path = qr_dir / "qr_01_detection.jpg"
    if cv2.imwrite(str(overlay_path), overlay):
        paths["detectionImage"] = str(overlay_path)

    qr_crop = qr_result.get("debugCrop")
    if isinstance(qr_crop, np.ndarray) and qr_crop.size:
        crop_path = qr_dir / "qr_02_crop.jpg"
        if cv2.imwrite(str(crop_path), qr_crop):
            paths["cropImage"] = str(crop_path)

    return paths


def save_parser_qr_overlay(
    fields_debug_path: str | Path,
    qr_result: dict[str, Any],
    card_size: tuple[int, int],
    output_dir: str | Path,
    output_name: str = "fields_parser_qr_debug.jpg",
) -> str | None:
    """Thêm QR vào ảnh khoanh field của parser để kiểm tra cùng một chỗ."""
    source_path = Path(fields_debug_path)
    image = cv2.imread(str(source_path))
    if image is None:
        return None
    overlay = draw_qr_overlay(
        image,
        qr_result,
        source_size=card_size,
    )
    output_path = Path(output_dir) / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), overlay):
        return None
    return str(output_path)

