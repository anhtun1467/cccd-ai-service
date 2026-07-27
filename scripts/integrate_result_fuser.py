from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

pipeline_path = (
    ROOT_DIR
    / "app"
    / "services"
    / "ocr_pipeline.py"
)

backup_path = (
    ROOT_DIR
    / "app"
    / "services"
    / "ocr_pipeline_before_result_fuser.py"
)

content = pipeline_path.read_text(
    encoding="utf-8-sig"
)

# ---------------------------------------------------------
# 1. T?o b?n sao luu
# ---------------------------------------------------------

if not backup_path.exists():
    backup_path.write_text(
        content,
        encoding="utf-8",
    )

# ---------------------------------------------------------
# 2. Thêm import fuse_ocr_data
# ---------------------------------------------------------

import_marker = (
    "from app.modules.ocr.line_merger "
    "import OCRLineMerger\n"
)

new_import = (
    "from app.modules.ocr.result_fuser "
    "import fuse_ocr_data\n"
)

if new_import not in content:
    if import_marker not in content:
        raise RuntimeError(
            "Không t́m th?y v? trí d? thêm import "
            "fuse_ocr_data."
        )

    content = content.replace(
        import_marker,
        import_marker + new_import,
        1,
    )

# ---------------------------------------------------------
# 3. Thay logic merge cu b?ng fuse_ocr_data
# ---------------------------------------------------------

old_merge_block = '''        merged_data = self.merge_structured_data(
            field_data=field_data,
            full_card_data=full_card_data,
        )

        validation_result = self.validator.validate(
            merged_data
        )
'''

new_merge_block = '''        raw_text_for_fusion = self.make_json_safe(
            full_ocr_result.get(
                "normalizedText",
                [],
            )
        )

        if isinstance(
            raw_text_for_fusion,
            str,
        ):
            raw_text_for_fusion = (
                raw_text_for_fusion.splitlines()
            )

        if not isinstance(
            raw_text_for_fusion,
            list,
        ):
            raw_text_for_fusion = []

        merged_data, data_sources = fuse_ocr_data(
            full_card_data=full_card_data,
            field_data=field_data,
            raw_text=raw_text_for_fusion,
        )

        validation_result = self.validator.validate(
            merged_data
        )
'''

if old_merge_block not in content:
    if "merged_data, data_sources = fuse_ocr_data(" not in content:
        raise RuntimeError(
            "Không t́m th?y block merge_structured_data "
            "c?n thay th?."
        )
else:
    content = content.replace(
        old_merge_block,
        new_merge_block,
        1,
    )

# ---------------------------------------------------------
# 4. Dùng l?i raw_text_for_fusion
# ---------------------------------------------------------

old_normalized_text_block = '''        normalized_text = self.make_json_safe(
            full_ocr_result.get(
                "normalizedText",
                [],
            )
        )
'''

new_normalized_text_block = '''        normalized_text = raw_text_for_fusion
'''

if old_normalized_text_block in content:
    content = content.replace(
        old_normalized_text_block,
        new_normalized_text_block,
        1,
    )

# ---------------------------------------------------------
# 5. Thay resolve_data_sources() b?ng data_sources
# ---------------------------------------------------------

old_sources_block = '''                "dataSources": (
                    self.resolve_data_sources(
                        field_data=field_data,
                        full_card_data=full_card_data,
                    )
                ),
'''

new_sources_block = '''                "dataSources": self.make_json_safe(
                    data_sources
                ),
'''

if old_sources_block not in content:
    if '"dataSources": self.make_json_safe(' not in content:
        raise RuntimeError(
            "Không t́m th?y block dataSources "
            "c?n thay th?."
        )
else:
    content = content.replace(
        old_sources_block,
        new_sources_block,
        1,
    )

pipeline_path.write_text(
    content,
    encoding="utf-8",
)

print("=" * 68)
print("TÍCH H?P RESULT_FUSER VÀO OCR PIPELINE THÀNH CÔNG")
print("=" * 68)
print(f"Pipeline: {pipeline_path}")
print(f"Backup:   {backup_path}")
