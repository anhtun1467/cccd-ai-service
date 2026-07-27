from __future__ import annotations

from pathlib import Path


FILES = (
    Path("tests/ocr/test_pipeline_result_fuser.py"),
    Path("tests/ocr/test_result_fuser.py"),
    Path("tests/ocr/test_text_normalizer.py"),
)


def decode_file(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()

    encodings = (
        "utf-8-sig",
        "utf-8",
        "cp1258",
        "cp1252",
        "latin-1",
    )

    for encoding in encodings:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(
        "unknown",
        raw,
        0,
        len(raw),
        f"Không thể xác định encoding của {path}",
    )


def remove_markdown_fences(content: str) -> str:
    lines = content.splitlines()

    while lines and lines[0].strip() in {
        "```python",
        "```py",
        "```",
    }:
        lines.pop(0)

    while lines and lines[-1].strip() == "```":
        lines.pop()

    return "\n".join(lines).strip() + "\n"


def replace_corrupted_print_lines(
    path: Path,
    content: str,
) -> str:
    lines = content.splitlines()
    fixed_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        if (
            path.name == "test_pipeline_result_fuser.py"
            and stripped.startswith("print(")
            and "RESULT_FUSER" in stripped.upper()
        ):
            indentation = line[: len(line) - len(line.lstrip())]
            fixed_lines.append(
                f'{indentation}print("TEST TÍCH HỢP RESULT_FUSER ĐÃ PASS")'
            )
            continue

        if (
            path.name == "test_result_fuser.py"
            and stripped.startswith("print(")
            and "result_fuser" in stripped.lower()
        ):
            indentation = line[: len(line) - len(line.lstrip())]
            fixed_lines.append(
                f'{indentation}print("Tất cả test result_fuser đã PASS.")'
            )
            continue

        fixed_lines.append(line)

    return "\n".join(fixed_lines).rstrip() + "\n"


def main() -> None:
    for path in FILES:
        if not path.exists():
            print(f"[SKIP] Không tồn tại: {path}")
            continue

        content, detected_encoding = decode_file(path)

        content = remove_markdown_fences(content)
        content = replace_corrupted_print_lines(
            path=path,
            content=content,
        )

        path.write_text(
            content,
            encoding="utf-8",
            newline="\n",
        )

        print(
            f"[FIXED] {path} | "
            f"{detected_encoding} -> utf-8"
        )


if __name__ == "__main__":
    main()
