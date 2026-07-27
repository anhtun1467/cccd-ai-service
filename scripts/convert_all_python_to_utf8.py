from __future__ import annotations

from pathlib import Path


ROOTS = (
    Path("app"),
    Path("scripts"),
    Path("tests"),
)

IGNORED_PARTS = {
    "venv",
    ".venv",
    ".git",
    "__pycache__",
    ".pytest_cache",
    "migration_backups",
    "encoding_backup",
    "encoding_fix_backup",
}

FALLBACK_ENCODINGS = (
    "cp1258",
    "cp1252",
    "latin-1",
)


def is_ignored(path: Path) -> bool:
    return any(
        part in IGNORED_PARTS
        or part.startswith("encoding_backup_app_")
        for part in path.parts
    )


def decode_source(raw: bytes) -> tuple[str, str]:
    # UTF-8 hoặc UTF-8 BOM
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            pass

    # Các encoding Windows thường gặp
    for encoding in FALLBACK_ENCODINGS:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            pass

    raise UnicodeDecodeError(
        "unknown",
        raw,
        0,
        len(raw),
        "Không thể giải mã file",
    )


def clean_markdown_fences(content: str) -> str:
    lines = content.splitlines()

    cleaned = [
        line
        for line in lines
        if line.strip() not in {
            "```",
            "```python",
            "```py",
        }
    ]

    return "\n".join(cleaned).rstrip() + "\n"


def main() -> None:
    converted = 0
    already_utf8 = 0
    failed = 0

    for root in ROOTS:
        if not root.exists():
            continue

        for path in sorted(root.rglob("*.py")):
            if is_ignored(path):
                continue

            raw = path.read_bytes()

            try:
                content, detected_encoding = decode_source(raw)
            except UnicodeDecodeError as exc:
                failed += 1
                print(f"[FAILED] {path}: {exc}")
                continue

            content = clean_markdown_fences(content)

            # Luôn lưu lại chuẩn UTF-8 không BOM
            path.write_text(
                content,
                encoding="utf-8",
                newline="\n",
            )

            if detected_encoding in {"utf-8", "utf-8-sig"}:
                already_utf8 += 1
                print(f"[UTF-8]     {path}")
            else:
                converted += 1
                print(
                    f"[CONVERTED] {path}: "
                    f"{detected_encoding} -> utf-8"
                )

    print()
    print("=" * 65)
    print(f"Đã chuyển encoding : {converted}")
    print(f"Đã là UTF-8        : {already_utf8}")
    print(f"Không đọc được     : {failed}")
    print("=" * 65)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
