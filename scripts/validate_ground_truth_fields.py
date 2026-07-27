from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = (
    "idNumber",
    "fullName",
    "dateOfBirth",
    "gender",
    "nationality",
    "placeOfOrigin",
    "placeOfResidence",
    "dateOfExpiry",
)

OLD_FIELDS = (
    "dateOfIssue",
    "issueDate",
    "expiryDate",
    "validUntil",
)


def extract_fields(data: dict[str, Any]) -> dict[str, Any]:
    """
    Hỗ trợ cả hai dạng:

    {
        "fields": {...}
    }

    và:

    {
        "idNumber": "...",
        ...
    }
    """
    fields = data.get("fields")

    if isinstance(fields, dict):
        return fields

    return data


def validate_json(path: Path) -> list[str]:
    errors: list[str] = []

    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception as exception:
        return [
            f"JSON không hợp lệ: {exception}"
        ]

    if not isinstance(data, dict):
        return ["Nội dung gốc phải là JSON object."]

    fields = extract_fields(data)

    for field_name in REQUIRED_FIELDS:
        if field_name not in fields:
            errors.append(
                f"Thiếu trường: {field_name}"
            )

    for old_field in OLD_FIELDS:
        if old_field in fields:
            errors.append(
                f"Còn trường cũ: {old_field}"
            )

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Kiểm tra Ground Truth có đủ "
            "8 trường CCCD hay không."
        )
    )

    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path(
            "datasets/source/ground_truth"
        ),
    )

    args = parser.parse_args()

    directory: Path = args.directory

    if not directory.exists():
        print(
            f"Chưa tồn tại thư mục: {directory}"
        )
        raise SystemExit(1)

    json_files = sorted(
        directory.rglob("*.json")
    )

    if not json_files:
        print(
            f"Không tìm thấy JSON trong: {directory}"
        )
        raise SystemExit(1)

    invalid_count = 0

    for path in json_files:
        errors = validate_json(path)

        if not errors:
            print(f"[PASS] {path}")
            continue

        invalid_count += 1
        print(f"[FAIL] {path}")

        for error in errors:
            print(f"       - {error}")

    print("=" * 72)
    print(f"Tổng JSON : {len(json_files)}")
    print(f"Hợp lệ    : {len(json_files) - invalid_count}")
    print(f"Không hợp lệ: {invalid_count}")
    print("=" * 72)

    if invalid_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
