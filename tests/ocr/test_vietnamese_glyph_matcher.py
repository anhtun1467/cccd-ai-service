from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from app.modules.ocr.glyph_matcher import (
    DEFAULT_ATLAS_PATH,
    DEFAULT_MANIFEST_PATH,
    VietnameseGlyphLibrary,
    VietnameseGlyphMatcher,
)
from app.modules.ocr.vietnamese_charset import (
    TEMPLATE_CHARACTERS,
    visual_candidates,
)


REQUESTED_SAMPLE = "aăâáàãạấầậẫắặ"


def _render_from_atlas(
    library: VietnameseGlyphLibrary,
    text: str,
    output_path: Path,
) -> tuple[int, int]:
    glyph_width = 64
    glyph_gap = 8
    space_width = 26
    width = sum(
        space_width if character.isspace() else glyph_width + glyph_gap
        for character in text
    ) + 12
    height = 76
    foreground = np.zeros((height, width), dtype=bool)
    x = 6
    for character in text:
        if character.isspace():
            x += space_width
            continue
        template = library.entries_for(character)[0].mask
        foreground[6:70, x:x + glyph_width] |= template
        x += glyph_width + glyph_gap

    image = np.full((height, width), 255, dtype=np.uint8)
    image[foreground] = 0
    Image.fromarray(image).save(output_path)
    return width, height


def test_atlas_is_complete_and_checksum_is_valid() -> None:
    library = VietnameseGlyphLibrary()

    assert library.available is True
    assert library.error is None
    assert library.character_count == len(TEMPLATE_CHARACTERS)
    assert library.manifest["uniqueCharacterCount"] == len(
        TEMPLATE_CHARACTERS
    )
    assert library.manifest["templateCount"] >= len(TEMPLATE_CHARACTERS) * 2


def test_requested_diacritics_rank_their_own_glyph_first() -> None:
    library = VietnameseGlyphLibrary()
    matcher = VietnameseGlyphMatcher(library=library)

    for character in REQUESTED_SAMPLE + REQUESTED_SAMPLE.upper():
        query = library.entries_for(character)[0].mask
        ranked = matcher.rank_candidates(
            query,
            visual_candidates(character, "fullName"),
        )
        assert ranked
        assert ranked[0][0] == character


def test_matcher_recovers_missing_marks_without_a_dictionary(
    tmp_path: Path,
) -> None:
    library = VietnameseGlyphLibrary()
    matcher = VietnameseGlyphMatcher(
        library=library,
        minimum_character_height=8,
    )
    image_path = tmp_path / "name.png"
    width, height = _render_from_atlas(
        library,
        "BÙI THỊ DUYÊN",
        image_path,
    )
    box = SimpleNamespace(
        text="BUI THI DUYEN",
        box=[[0, 0], [width, 0], [width, height], [0, height]],
    )

    overrides, report = matcher.refine_ocr_boxes(
        image_path=image_path,
        text_boxes=[box],
        field_name="fullName",
    )

    assert overrides[0] == "BÙI THỊ DUYÊN"
    assert report["available"] is True
    assert report["applied"] is True
    assert report["coverage"] > 0.90
    assert report["corrections"]


def test_numeric_field_can_verify_letter_digit_confusion(
    tmp_path: Path,
) -> None:
    library = VietnameseGlyphLibrary()
    matcher = VietnameseGlyphMatcher(
        library=library,
        minimum_character_height=8,
    )
    image_path = tmp_path / "id.png"
    width, height = _render_from_atlas(
        library,
        "0123456789",
        image_path,
    )
    box = SimpleNamespace(
        text="O123456789",
        box=[[0, 0], [width, 0], [width, height], [0, height]],
    )

    overrides, report = matcher.refine_ocr_boxes(
        image_path=image_path,
        text_boxes=[box],
        field_name="idNumber",
    )

    assert overrides[0] == "0123456789"
    assert report["corrections"][0]["from"] == "O"
    assert report["corrections"][0]["to"] == "0"


def test_report_only_mode_never_changes_ocr_text(
    tmp_path: Path,
) -> None:
    library = VietnameseGlyphLibrary()
    matcher = VietnameseGlyphMatcher(
        library=library,
        minimum_character_height=8,
        auto_correct=False,
    )
    image_path = tmp_path / "report-only.png"
    width, height = _render_from_atlas(library, "BÙI", image_path)
    box = SimpleNamespace(
        text="BUI",
        box=[[0, 0], [width, 0], [width, height], [0, height]],
    )

    overrides, report = matcher.refine_ocr_boxes(
        image_path=image_path,
        text_boxes=[box],
        field_name="fullName",
    )

    assert overrides == {}
    assert report["autoCorrect"] is False
    assert report["skippedReason"] == "REPORT_ONLY"
    assert any(
        item["decision"] == "REPORT_ONLY"
        for item in report["reviewCandidates"]
    )


def test_low_resolution_box_is_never_auto_corrected(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "tiny.png"
    Image.new("L", (60, 8), color=255).save(image_path)
    box = SimpleNamespace(
        text="A",
        box=[[0, 0], [60, 0], [60, 8], [0, 8]],
    )

    overrides, report = VietnameseGlyphMatcher().refine_ocr_boxes(
        image_path=image_path,
        text_boxes=[box],
        field_name="fullName",
    )

    assert overrides == {}
    assert report["applied"] is False
    assert report["skippedReason"] == "NO_USABLE_TEXT_BOX"


def test_checksum_mismatch_disables_only_the_optional_library(
    tmp_path: Path,
) -> None:
    atlas_path = tmp_path / "atlas.npz"
    manifest_path = tmp_path / "manifest.json"
    atlas_path.write_bytes(DEFAULT_ATLAS_PATH.read_bytes())
    manifest = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["assetSha256"] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )

    library = VietnameseGlyphLibrary(
        atlas_path=atlas_path,
        manifest_path=manifest_path,
    )

    assert library.available is False
    assert "checksum" in str(library.error).lower()
