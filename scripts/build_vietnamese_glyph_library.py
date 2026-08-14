from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.modules.ocr.vietnamese_charset import (  # noqa: E402
    TEMPLATE_CHARACTERS,
    VIETNAMESE_LETTERS_LOWER,
    VIETNAMESE_LETTERS_UPPER,
)


DEFAULT_FONT_CANDIDATES = (
    # Windows - ưu tiên các font gần kiểu chữ dùng trên giấy tờ.
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("C:/Windows/Fonts/arialbd.ttf"),
    Path("C:/Windows/Fonts/tahoma.ttf"),
    Path("C:/Windows/Fonts/tahomabd.ttf"),
    Path("C:/Windows/Fonts/segoeui.ttf"),
    Path("C:/Windows/Fonts/seguisb.ttf"),
    # Linux/macOS - dùng để tạo atlas có sẵn trong gói bàn giao.
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"),
    Path("/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf"),
    Path("/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf"),
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tạo atlas mẫu ảnh cho toàn bộ chữ/số tiếng Việt dùng trong OCR."
        )
    )
    parser.add_argument(
        "--font",
        action="append",
        default=[],
        help="Đường dẫn font TTF/OTF; có thể truyền nhiều lần.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "app" / "modules" / "ocr" / "glyphs"),
    )
    parser.add_argument("--font-size", type=int, default=72)
    parser.add_argument("--canvas-size", type=int, default=64)
    return parser.parse_args()


def choose_fonts(requested: list[str]) -> list[Path]:
    candidates = (
        [Path(value).expanduser() for value in requested]
        if requested
        else list(DEFAULT_FONT_CANDIDATES)
    )
    selected: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate.is_file():
            continue
        key = str(candidate.resolve()).casefold()
        if key in seen:
            continue
        seen.add(key)
        selected.append(candidate)
    if not selected:
        raise FileNotFoundError(
            "Không tìm thấy font Unicode tiếng Việt. "
            "Hãy truyền --font C:\\Windows\\Fonts\\arial.ttf"
        )
    return selected


def render_character(
    character: str,
    font: ImageFont.FreeTypeFont,
    canvas_size: int,
) -> tuple[np.ndarray, float]:
    ascent, descent = font.getmetrics()
    padding = max(8, int(round(font.size * 0.20)))
    advance = max(1, int(round(font.getlength(character))))
    raw_width = max(advance + padding * 2, font.size + padding * 2)
    raw_height = ascent + descent + padding * 2

    image = Image.new("L", (raw_width, raw_height), color=0)
    draw = ImageDraw.Draw(image)
    baseline = padding + ascent
    draw.text(
        (padding, baseline),
        character,
        font=font,
        fill=255,
        anchor="ls",
    )
    mask = np.asarray(image, dtype=np.uint8) >= 48
    rows, columns = np.where(mask)
    if rows.size == 0 or columns.size == 0:
        raise ValueError(f"Font không render được U+{ord(character):04X}")

    cropped = mask[
        int(rows.min()):int(rows.max()) + 1,
        int(columns.min()):int(columns.max()) + 1,
    ]
    height, width = cropped.shape
    aspect_ratio = float(width / max(height, 1))
    margin = 4
    available = canvas_size - margin * 2
    scale = min(available / width, available / height)
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    resized = Image.fromarray(cropped.astype(np.uint8) * 255).resize(
        (target_width, target_height),
        Image.Resampling.NEAREST,
    )
    normalized = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    x = (canvas_size - target_width) // 2
    y = (canvas_size - target_height) // 2
    normalized[
        y:y + target_height,
        x:x + target_width,
    ] = (np.asarray(resized, dtype=np.uint8) > 0).astype(np.uint8)
    return normalized, aspect_ratio


def build_atlas(
    fonts: list[Path],
    output_dir: Path,
    font_size: int,
    canvas_size: int,
) -> tuple[Path, Path]:
    if font_size < 24:
        raise ValueError("font-size phải từ 24 trở lên")
    if canvas_size < 32:
        raise ValueError("canvas-size phải từ 32 trở lên")

    output_dir.mkdir(parents=True, exist_ok=True)
    atlas_path = output_dir / "vietnamese_glyph_templates.npz"
    manifest_path = output_dir / "manifest.json"

    templates: list[np.ndarray] = []
    characters: list[str] = []
    font_names: list[str] = []
    aspect_ratios: list[float] = []
    accepted_fonts: list[Path] = []

    for font_path in fonts:
        font = ImageFont.truetype(str(font_path), size=font_size)
        font_templates: list[np.ndarray] = []
        font_aspects: list[float] = []
        try:
            for character in TEMPLATE_CHARACTERS:
                template, aspect_ratio = render_character(
                    character=character,
                    font=font,
                    canvas_size=canvas_size,
                )
                font_templates.append(template)
                font_aspects.append(aspect_ratio)
        except ValueError as error:
            print(
                f"Bỏ qua font {font_path.name}: {error}",
                file=sys.stderr,
            )
            continue

        accepted_fonts.append(font_path)
        templates.extend(font_templates)
        aspect_ratios.extend(font_aspects)
        characters.extend(TEMPLATE_CHARACTERS)
        font_names.extend([font_path.name] * len(TEMPLATE_CHARACTERS))

    if not accepted_fonts:
        raise ValueError("Không có font nào hỗ trợ đầy đủ bảng tiếng Việt")

    np.savez_compressed(
        atlas_path,
        templates=np.stack(templates).astype(np.uint8),
        characters=np.asarray(characters),
        font_names=np.asarray(font_names),
        aspect_ratios=np.asarray(aspect_ratios, dtype=np.float32),
    )
    asset_hash = hashlib.sha256(atlas_path.read_bytes()).hexdigest()

    manifest = {
        "schemaVersion": 1,
        "libraryVersion": "2026.08.14",
        "normalization": "Unicode NFC",
        "canvasSize": canvas_size,
        "fontSize": font_size,
        "fontFiles": [path.name for path in accepted_fonts],
        "uniqueCharacterCount": len(TEMPLATE_CHARACTERS),
        "templateCount": len(templates),
        "lowercaseVietnameseLetterCount": len(VIETNAMESE_LETTERS_LOWER),
        "uppercaseVietnameseLetterCount": len(VIETNAMESE_LETTERS_UPPER),
        "characters": TEMPLATE_CHARACTERS,
        "assetFile": atlas_path.name,
        "assetSha256": asset_hash,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return atlas_path, manifest_path


def main() -> None:
    args = parse_args()
    fonts = choose_fonts(args.font)
    atlas_path, manifest_path = build_atlas(
        fonts=fonts,
        output_dir=Path(args.output_dir),
        font_size=args.font_size,
        canvas_size=args.canvas_size,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"Fonts: {', '.join(manifest['fontFiles'])}")
    print(f"Atlas: {atlas_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
