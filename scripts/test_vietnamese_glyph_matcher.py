from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.modules.ocr.glyph_matcher import (  # noqa: E402
    VietnameseGlyphMatcher,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "So sánh một OCR box/crop với thư viện chữ số tiếng Việt."
        )
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--text", required=True, help="Kết quả EasyOCR ban đầu")
    parser.add_argument(
        "--field",
        required=True,
        choices=(
            "idNumber",
            "fullName",
            "dateOfBirth",
            "gender",
            "nationality",
            "placeOfOrigin",
            "placeOfResidence",
            "dateOfExpiry",
        ),
    )
    parser.add_argument(
        "--box",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Tọa độ box; bỏ qua để dùng toàn bộ ảnh.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

    with Image.open(image_path) as image:
        width, height = image.size

    if args.box:
        x1, y1, x2, y2 = args.box
    else:
        x1, y1, x2, y2 = 0.0, 0.0, float(width), float(height)

    text_box = SimpleNamespace(
        text=args.text,
        box=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
    )
    overrides, report = VietnameseGlyphMatcher().refine_ocr_boxes(
        image_path=image_path,
        text_boxes=[text_box],
        field_name=args.field,
    )
    result = {
        "originalText": args.text,
        "refinedText": overrides.get(0, args.text),
        "changed": 0 in overrides,
        "glyphMatch": report,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
