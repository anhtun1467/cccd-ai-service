from __future__ import annotations

import re
from pathlib import Path


PATH = Path("app/modules/ocr/result_fuser.py")

content = PATH.read_text(encoding="utf-8")


NEW_NORMALIZE_GENDER = r'''
def normalize_gender(value: str | None) -> str | None:
    """
    Chuẩn hóa giới tính.

    Quy tắc:
        Nam / Male    -> Nam
        Nu / Nữ / Female -> Nu
        Viet Nam      -> None

    Chỉ nhận token độc lập, tuyệt đối không tìm substring "nam"
    bên trong "Viet Nam".
    """

    if not value:
        return None

    text = remove_accents(str(value)).lower()

    text = re.sub(
        r"[^a-z\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    tokens = set(text.split())

    if "nu" in tokens or "female" in tokens:
        return "Nu"

    if "nam" in tokens or "male" in tokens:
        return "Nam"

    return None
'''.strip()


NEW_RECOVER_GENDER = r'''
def recover_gender(
    raw_text: list[str] | None,
) -> str | None:
    """
    Phục hồi giới tính từ đúng dòng chứa nhãn Giới tính / Sex.

    Ví dụ:
        Gioi tinh / Sex: Nu Quoc tich / Nationality: Viet Nam

    Kết quả:
        Nu

    Phần Quốc tịch bị cắt bỏ trước khi chuẩn hóa để chữ
    "Nam" trong "Viet Nam" không gây nhận sai.
    """

    if not raw_text:
        return None

    for raw_line in raw_text:
        if not raw_line:
            continue

        line = remove_accents(
            str(raw_line)
        )

        normalized_line = re.sub(
            r"\s+",
            " ",
            line,
        ).strip()

        # Chỉ xét các dòng có nhãn giới tính.
        if not re.search(
            r"\b(?:gioi\s*tinh|sex)\b",
            normalized_line,
            flags=re.IGNORECASE,
        ):
            continue

        # Cắt bỏ toàn bộ phần quốc tịch phía sau.
        gender_section = re.split(
            r"\b(?:quoc\s*tich|nationality)\b",
            normalized_line,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]

        # Lấy nội dung sau nhãn Sex: nếu có.
        sex_match = re.search(
            r"\bsex\b\s*[:/\-]?\s*(.*)$",
            gender_section,
            flags=re.IGNORECASE,
        )

        if sex_match:
            gender_value = sex_match.group(1)
        else:
            # Fallback lấy sau nhãn Giới tính.
            gender_match = re.search(
                r"\bgioi\s*tinh\b"
                r"\s*[:/\-]?\s*"
                r"(.*)$",
                gender_section,
                flags=re.IGNORECASE,
            )

            if not gender_match:
                continue

            gender_value = gender_match.group(1)

        # Xóa nhãn Sex còn sót nếu OCR ghép dạng:
        # Gioi tinh / Sex: Nu
        gender_value = re.sub(
            r"^\s*/?\s*sex\s*[:/\-]?\s*",
            "",
            gender_value,
            flags=re.IGNORECASE,
        )

        gender = normalize_gender(
            gender_value
        )

        if gender:
            return gender

    return None
'''.strip()


def replace_function(
    source: str,
    function_name: str,
    replacement: str,
) -> str:
    pattern = re.compile(
        rf"^def {re.escape(function_name)}\(.*?"
        rf"(?=^def |\Z)",
        flags=re.MULTILINE | re.DOTALL,
    )

    match = pattern.search(source)

    if not match:
        raise RuntimeError(
            f"Không tìm thấy hàm: {function_name}"
        )

    return (
        source[:match.start()]
        + replacement
        + "\n\n\n"
        + source[match.end():]
    )


content = replace_function(
    content,
    "normalize_gender",
    NEW_NORMALIZE_GENDER,
)

content = replace_function(
    content,
    "recover_gender",
    NEW_RECOVER_GENDER,
)

PATH.write_text(
    content,
    encoding="utf-8",
    newline="\n",
)

print("[OK] Đã sửa normalize_gender()")
print("[OK] Đã sửa recover_gender()")
print(f"[OK] File: {PATH}")
