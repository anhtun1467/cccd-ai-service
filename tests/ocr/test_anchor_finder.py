from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT_DIR),
    )


from app.modules.ocr.layout.anchor_finder import (
    AnchorFinder,
    calculate_text_similarity,
    compact_anchor_text,
    normalize_anchor_text,
    remove_accents,
)
from app.modules.ocr.layout.text_box import (
    LayoutTextBox,
)


def create_box(
    text: str,
    left: float,
    top: float,
    right: float,
    bottom: float,
    confidence: float = 0.90,
) -> LayoutTextBox:
    return LayoutTextBox.from_rectangle(
        text=text,
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        confidence=confidence,
    )


def build_boxes() -> list[LayoutTextBox]:
    return [
        create_box(
            text="Số / No:",
            left=317,
            top=309,
            right=421,
            bottom=345,
            confidence=0.95,
        ),
        create_box(
            text="001095014159",
            left=424,
            top=301,
            right=780,
            bottom=355,
            confidence=0.99,
        ),
        create_box(
            text="Họ và tên / Full name:",
            left=318,
            top=358,
            right=562,
            bottom=390,
            confidence=0.94,
        ),
        create_box(
            text="NGUYEN HOANG NAM",
            left=314,
            top=394,
            right=739,
            bottom=436,
            confidence=0.98,
        ),
        create_box(
            text="Ngày sinh / Date of birth:",
            left=318,
            top=440,
            right=600,
            bottom=474,
            confidence=0.93,
        ),
        create_box(
            text="24/03/1995",
            left=611,
            top=439,
            right=789,
            bottom=477,
            confidence=0.96,
        ),
        create_box(
            text="Giới tính / Sex:",
            left=318,
            top=478,
            right=493,
            bottom=510,
            confidence=0.92,
        ),
        create_box(
            text="Nam",
            left=503,
            top=475,
            right=579,
            bottom=511,
            confidence=0.99,
        ),
        create_box(
            text="Quốc tịch / Nationality:",
            left=591,
            top=475,
            right=841,
            bottom=514,
            confidence=0.91,
        ),
        create_box(
            text="Việt Nam",
            left=857,
            top=473,
            right=1001,
            bottom=513,
            confidence=0.97,
        ),
        create_box(
            text="Quê quán / Place of origin:",
            left=317,
            top=523,
            right=615,
            bottom=559,
            confidence=0.90,
        ),
        create_box(
            text="Bạch Đằng, Hạ Lý, Hải Phòng",
            left=317,
            top=565,
            right=850,
            bottom=605,
            confidence=0.88,
        ),
    ]


def test_remove_accents() -> None:
    result = remove_accents(
        "Họ và tên Quốc tịch"
    )

    assert result == (
        "Ho va ten Quoc tich"
    )


def test_normalize_anchor_text() -> None:
    result = normalize_anchor_text(
        "  Họ-và-tên / Full name:  "
    )

    assert result == (
        "ho va ten full name"
    )


def test_compact_anchor_text() -> None:
    result = compact_anchor_text(
        "Ngày sinh"
    )

    assert result == "ngaysinh"


def test_exact_similarity() -> None:
    result = calculate_text_similarity(
        "Họ và tên",
        "Ho va ten",
    )

    assert result == 1.0


def test_joined_text_similarity() -> None:
    result = calculate_text_similarity(
        "Hovaten",
        "Họ và tên",
    )

    assert result == 1.0


def test_ocr_error_similarity() -> None:
    result = calculate_text_similarity(
        "Ngay sinb",
        "Ngày sinh",
    )

    assert result >= 0.80


def test_find_anchor() -> None:
    finder = AnchorFinder(
        build_boxes()
    )

    anchor = finder.find_anchor(
        anchor_name="fullName",
        aliases=[
            "Họ và tên",
            "Ho va ten",
            "Full name",
        ],
    )

    assert anchor is not None

    assert anchor.box.text == (
        "Họ và tên / Full name:"
    )

    assert anchor.similarity >= 0.60


def test_find_right_date() -> None:
    finder = AnchorFinder(
        build_boxes()
    )

    result = finder.find_value(
        anchor_name="dateOfBirth",
        aliases=[
            "Ngày sinh",
            "Ngay sinh",
            "Date of birth",
        ],
        preferred_direction="right",
        fallback_direction="below",
    )

    assert result is not None

    assert result.candidate.text == (
        "24/03/1995"
    )

    assert result.direction == "right"


def test_find_right_gender() -> None:
    finder = AnchorFinder(
        build_boxes()
    )

    result = finder.find_value(
        anchor_name="gender",
        aliases=[
            "Giới tính",
            "Gioi tinh",
            "Sex",
        ],
        preferred_direction="right",
    )

    assert result is not None

    assert result.candidate.text == "Nam"


def test_find_right_nationality() -> None:
    finder = AnchorFinder(
        build_boxes()
    )

    result = finder.find_value(
        anchor_name="nationality",
        aliases=[
            "Quốc tịch",
            "Quoc tich",
            "Nationality",
        ],
        preferred_direction="right",
    )

    assert result is not None

    assert result.candidate.text == (
        "Việt Nam"
    )


def test_find_below_name() -> None:
    finder = AnchorFinder(
        build_boxes()
    )

    result = finder.find_value(
        anchor_name="fullName",
        aliases=[
            "Họ và tên",
            "Ho va ten",
            "Full name",
        ],
        preferred_direction="below",
        fallback_direction="right",
    )

    assert result is not None

    assert result.candidate.text == (
        "NGUYEN HOANG NAM"
    )

    assert result.direction == "below"


def test_find_below_origin() -> None:
    finder = AnchorFinder(
        build_boxes()
    )

    result = finder.find_value(
        anchor_name="placeOfOrigin",
        aliases=[
            "Quê quán",
            "Que quan",
            "Place of origin",
        ],
        preferred_direction="below",
    )

    assert result is not None

    assert result.candidate.text == (
        "Bạch Đằng, Hạ Lý, Hải Phòng"
    )


def test_missing_anchor() -> None:
    finder = AnchorFinder(
        build_boxes()
    )

    result = finder.find_value(
        anchor_name="unknown",
        aliases=[
            "Không tồn tại",
        ],
    )

    assert result is None


def main() -> None:
    tests = (
        test_remove_accents,
        test_normalize_anchor_text,
        test_compact_anchor_text,
        test_exact_similarity,
        test_joined_text_similarity,
        test_ocr_error_similarity,
        test_find_anchor,
        test_find_right_date,
        test_find_right_gender,
        test_find_right_nationality,
        test_find_below_name,
        test_find_below_origin,
        test_missing_anchor,
    )

    passed = 0

    for test_function in tests:
        try:
            test_function()

            print(
                f"[PASS] "
                f"{test_function.__name__}"
            )

            passed += 1

        except Exception as error:
            print(
                f"[FAIL] "
                f"{test_function.__name__}: "
                f"{error}"
            )

    print("-" * 60)

    print(
        f"Kết quả: "
        f"{passed}/{len(tests)} "
        f"test thành công"
    )

    if passed != len(tests):
        raise AssertionError(
            "Có test Anchor Finder chưa đạt"
        )


if __name__ == "__main__":
    main()
