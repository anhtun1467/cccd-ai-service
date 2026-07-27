from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path.cwd()

ALLOWED_SUFFIXES = {
    ".py",
    ".json",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".html",
    ".css",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
}

IGNORED_DIRECTORIES = {
    ".git",
    ".idea",
    ".vscode",
    ".pytest_cache",
    "__pycache__",
    "venv",
    ".venv",
    "env",
    "node_modules",
    "dist",
    "build",
    "coverage",
    "storage",
}

BACKUP_ROOT = (
    PROJECT_ROOT
    / "migration_backups"
    / datetime.now().strftime("%Y%m%d_%H%M%S")
)

REPLACEMENTS: list[tuple[str, str]] = [
    # camelCase
    ("dateOfExpiry", "dateOfExpiry"),
    ("dateOfExpiry", "dateOfExpiry"),
    ("dateOfExpiry", "dateOfExpiry"),
    ("dateOfExpiry", "dateOfExpiry"),

    # snake_case
    ("date_of_expiry", "date_of_expiry"),
    ("date_of_expiry", "date_of_expiry"),
    ("date_of_expiry", "date_of_expiry"),
    ("date_of_expiry", "date_of_expiry"),

    # UPPER_CASE constants
    ("DATE_OF_EXPIRY", "DATE_OF_EXPIRY"),
    ("DATE_OF_EXPIRY", "DATE_OF_EXPIRY"),
    ("DATE_OF_EXPIRY", "DATE_OF_EXPIRY"),
    ("DATE_OF_EXPIRY", "DATE_OF_EXPIRY"),

    # Nhãn tiếng Việt
    ("Có giá trị đến", "Có giá trị đến"),
    ("CÓ GIÁ TRỊ ĐẾN", "CÓ GIÁ TRỊ ĐẾN"),
    ("có giá trị đến", "có giá trị đến"),
    ("Co gia tri den", "Co gia tri den"),
    ("CO GIA TRI DEN", "CO GIA TRI DEN"),
    ("co gia tri den",
                "có giá trị đến", "co gia tri den"),

    # Nhãn tiếng Anh
    ("Date of expiry", "Date of expiry"),
    ("DATE OF EXPIRY", "DATE OF EXPIRY"),
    ("date of expiry",
                "valid until",
                "expiry date", "date of expiry"),
    ("Date of expiry", "Date of expiry"),
    ("Date of expiry", "Date of expiry"),
]


def should_process(path: Path) -> bool:
    if not path.is_file():
        return False

    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        return False

    relative_parts = path.relative_to(PROJECT_ROOT).parts

    return not any(
        directory in IGNORED_DIRECTORIES
        for directory in relative_parts
    )


def read_text_safely(path: Path) -> tuple[str, str] | None:
    encodings = (
        "utf-8",
        "utf-8-sig",
        "cp1258",
        "cp1252",
    )

    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding), encoding
        except UnicodeDecodeError:
            continue

    print(f"[SKIP ENCODING] {path}")
    return None


def backup_file(path: Path) -> None:
    relative_path = path.relative_to(PROJECT_ROOT)
    backup_path = BACKUP_ROOT / relative_path

    backup_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(path, backup_path)


def apply_basic_replacements(content: str) -> str:
    updated = content

    for old_value, new_value in REPLACEMENTS:
        updated = updated.replace(
            old_value,
            new_value,
        )

    return updated


def normalize_field_labels(content: str) -> str:
    """
    Bổ sung một số biến thể nhãn OCR thường gặp.

    Chỉ bổ sung khi file đã có nội dung liên quan đến dateOfExpiry
    hoặc date_of_expiry.
    """
    if (
        "dateOfExpiry" not in content
        and "date_of_expiry" not in content
    ):
        return content

    replacements = [
        (
            '"date of expiry",',
            '"date of expiry",\n'
            '                "valid until",\n'
            '                "expiry date",',
        ),
        (
            "'date of expiry',
                'valid until',
                'expiry date',",
            "'date of expiry',\n"
            "                'valid until',\n"
            "                'expiry date',",
        ),
        (
            '"co gia tri den",',
            '"co gia tri den",\n'
            '                "có giá trị đến",',
        ),
        (
            "'co gia tri den',
                'có giá trị đến',",
            "'co gia tri den',\n"
            "                'có giá trị đến',",
        ),
    ]

    updated = content

    for old_value, new_value in replacements:
        if (
            old_value in updated
            and new_value not in updated
        ):
            updated = updated.replace(
                old_value,
                new_value,
                1,
            )

    return updated


def migrate_file(path: Path) -> bool:
    loaded = read_text_safely(path)

    if loaded is None:
        return False

    original_content, _ = loaded

    updated_content = apply_basic_replacements(
        original_content
    )

    updated_content = normalize_field_labels(
        updated_content
    )

    if updated_content == original_content:
        return False

    backup_file(path)

    path.write_text(
        updated_content,
        encoding="utf-8",
        newline="\n",
    )

    print(f"[UPDATED] {path.relative_to(PROJECT_ROOT)}")
    return True


def scan_old_terms() -> list[tuple[Path, int, str]]:
    old_patterns = (
        "dateOfExpiry",
        "dateOfExpiry",
        "date_of_expiry",
        "date_of_expiry",
        "Có giá trị đến",
        "có giá trị đến",
        "Date of expiry",
        "date of expiry",
    )

    results: list[tuple[Path, int, str]] = []

    for path in PROJECT_ROOT.rglob("*"):
        if not should_process(path):
            continue

        loaded = read_text_safely(path)

        if loaded is None:
            continue

        content, _ = loaded

        for line_number, line in enumerate(
            content.splitlines(),
            start=1,
        ):
            if any(
                pattern in line
                for pattern in old_patterns
            ):
                results.append(
                    (
                        path.relative_to(PROJECT_ROOT),
                        line_number,
                        line.strip(),
                    )
                )

    return results


def scan_new_field() -> list[Path]:
    results: list[Path] = []

    for path in PROJECT_ROOT.rglob("*"):
        if not should_process(path):
            continue

        loaded = read_text_safely(path)

        if loaded is None:
            continue

        content, _ = loaded

        if (
            "dateOfExpiry" in content
            or "date_of_expiry" in content
            or "Có giá trị đến" in content
        ):
            results.append(
                path.relative_to(PROJECT_ROOT)
            )

    return sorted(set(results))


def main() -> None:
    if not (PROJECT_ROOT / "app").exists():
        raise RuntimeError(
            "Hãy chạy script tại thư mục gốc "
            "cccd-ai-service."
        )

    print("=" * 72)
    print("MIGRATE TRƯỜNG CÓ GIÁ TRỊ ĐẾN -> CÓ GIÁ TRỊ ĐẾN")
    print(f"Project: {PROJECT_ROOT}")
    print(f"Backup : {BACKUP_ROOT}")
    print("=" * 72)

    updated_count = 0

    for path in PROJECT_ROOT.rglob("*"):
        if not should_process(path):
            continue

        if migrate_file(path):
            updated_count += 1

    print()
    print("=" * 72)
    print(f"Số file đã sửa: {updated_count}")
    print("=" * 72)

    remaining_old_terms = scan_old_terms()

    if remaining_old_terms:
        print()
        print("[CẢNH BÁO] Vẫn còn tên trường cũ:")

        for path, line_number, line in remaining_old_terms:
            print(
                f"- {path}:{line_number}: {line}"
            )
    else:
        print()
        print("[OK] Không còn tên trường có giá trị đến cũ.")

    new_field_files = scan_new_field()

    print()
    print(
        f"Trường dateOfExpiry/date_of_expiry "
        f"xuất hiện trong {len(new_field_files)} file:"
    )

    for path in new_field_files:
        print(f"- {path}")

    print()
    print("HOÀN TẤT.")
    print(
        "Hãy chạy git diff và test trước khi commit."
    )


if __name__ == "__main__":
    main()
