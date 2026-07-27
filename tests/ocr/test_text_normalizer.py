from __future__ import annotations

import sys
from pathlib import Path


# ThÃªm thÆ° má»¥c gá»‘c cá»§a project vÃ o PYTHONPATH Ä‘á»ƒ import Ä‘Æ°á»£c package app.
ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from app.modules.ocr.text_normalizer import OCRTextNormalizer


def run_test(
    input_text: str,
    expected_text: str,
) -> bool:
    """
    Cháº¡y má»™t trÆ°á»ng há»£p kiá»ƒm thá»­ OCRTextNormalizer.
    """

    actual_text = OCRTextNormalizer.normalize(input_text)
    passed = actual_text == expected_text

    print("=" * 60)
    print("[PASS]" if passed else "[FAIL]")
    print(f"Input   : {input_text}")
    print(f"Expected: {expected_text}")
    print(f"Actual  : {actual_text}")

    return passed


def main() -> None:
    test_cases: list[tuple[str, str]] = [
        (
            "Vict Nana",
            "Viet Nam",
        ),
        (
            "Ha_Noi",
            "Ha Noi",
        ),
        (
            "24 / 03 / 1995",
            "24/03/1995",
        ),
        (
            "Hova ten / Full name:",
            "Ho va ten/Full name:",
        ),
        (
            "CONG_DAN",
            "CONG DAN",
        ),
        (
            "Gioitinh / Sex:",
            "Gioi tinh/Sex:",
        ),
        (
            "Ngay sinh / Date ofbirth:",
            "Ngay sinh/Date of birth:",
        ),
        (
            "Date ofDxpiry",
            "Date of Expiry",
        ),
        (
            "Freedom_Happiness",
            "Freedom - Happiness",
        ),
        (
            "Noi   thuong   tru",
            "Noi thuong tru",
        ),
    ]

    passed_count = 0

    for input_text, expected_text in test_cases:
        if run_test(input_text, expected_text):
            passed_count += 1

    total_count = len(test_cases)
    failed_count = total_count - passed_count

    print("=" * 60)
    print(f"Total : {total_count}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print("=" * 60)

    if failed_count > 0:
        raise AssertionError(
            f"CÃ³ {failed_count} trÆ°á»ng há»£p kiá»ƒm thá»­ chÆ°a Ä‘áº¡t."
        )

    print("Táº¥t cáº£ test OCRTextNormalizer Ä‘á»u thÃ nh cÃ´ng.")


if __name__ == "__main__":
    main()
