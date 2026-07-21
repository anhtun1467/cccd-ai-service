from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from app.modules.ocr.layout import (
    CCCDLayoutParser,
    LayoutTextBox,
)


def create_box(
    text: str,
    left: float,
    top: float,
    right: float,
    bottom: float,
    confidence: float = 0.95,
) -> LayoutTextBox:
    return LayoutTextBox.from_rectangle(
        text=text,
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        confidence=confidence,
    )


def build_sample_boxes() -> list[LayoutTextBox]:
    return [
        create_box(
            "Số / No:",
            317,
            309,
            421,
            345,
        ),
        create_box(
            "001095014159",
            424,
            301,
            780,
            355,
            0.99,
        ),
        create_box(
            "Họ và tên / Full name:",
            318,
            358,
            562,
            390,
        ),
        create_box(
            "NGUYEN HOANG NAM",
            314,
            394,
            739,
            436,
            0.98,
        ),
        create_box(
            "Ngày sinh / Date of birth:",
            318,
            440,
            600,
            474,
        ),
        create_box(
            "24/03/1995",
            611,
            439,
            789,
            477,
            0.97,
        ),
        create_box(
            "Giới tính / Sex:",
            318,
            478,
            493,
            510,
        ),
        create_box(
            "Nam",
            503,
            475,
            579,
            511,
            0.99,
        ),
        create_box(
            "Quốc tịch / Nationality:",
            591,
            475,
            841,
            514,
        ),
        create_box(
            "Việt Nam",
            857,
            473,
            1001,
            513,
            0.98,
        ),
        create_box(
            "Quê quán / Place of origin:",
            317,
            523,
            615,
            559,
        ),
        create_box(
            "Bạch Đằng, Hạ Lý, Hải Phòng",
            317,
            565,
            850,
            605,
            0.94,
        ),
        create_box(
            "Nơi thường trú / Place of residence:",
            317,
            615,
            760,
            651,
        ),
        create_box(
            "12 Lê Lợi, Ngô Quyền, Hải Phòng",
            317,
            659,
            900,
            701,
            0.93,
        ),
        create_box(
            "Có giá trị đến / Date of expiry:",
            317,
            715,
            710,
            750,
        ),
        create_box(
            "24/03/2035",
            720,
            713,
            900,
            752,
            0.96,
        ),
    ]


def test_parse_all_fields() -> None:
    parser = CCCDLayoutParser(
        build_sample_boxes()
    )

    result = parser.parse()

    assert result.id_number == "001095014159"

    assert result.full_name == (
        "NGUYEN HOANG NAM"
    )

    assert result.date_of_birth == (
        "24/03/1995"
    )

    assert result.gender == "Nam"

    assert result.nationality == (
        "Việt Nam"
    )

    assert result.place_of_origin == (
        "Bạch Đằng, Hạ Lý, Hải Phòng"
    )

    assert result.place_of_residence == (
        "12 Lê Lợi, Ngô Quyền, Hải Phòng"
    )

    assert result.date_of_expiry == (
        "24/03/2035"
    )


def test_parse_to_dict() -> None:
    parser = CCCDLayoutParser(
        build_sample_boxes()
    )

    result = parser.parse()

    data = result.to_dict(
        camel_case=True
    )

    assert data["idNumber"] == (
        "001095014159"
    )

    assert data["fullName"] == (
        "NGUYEN HOANG NAM"
    )

    assert data["dateOfBirth"] == (
        "24/03/1995"
    )

    assert data["placeOfResidence"] == (
        "12 Lê Lợi, Ngô Quyền, Hải Phòng"
    )


def test_parse_with_debug() -> None:
    parser = CCCDLayoutParser(
        build_sample_boxes()
    )

    debug_result = (
        parser.parse_with_debug()
    )

    data = debug_result.to_dict()

    assert (
        data["result"]["idNumber"]
        == "001095014159"
    )

    assert (
        data["matches"]["idNumber"]
        is not None
    )

    assert (
        data["matches"]["fullName"]
        is not None
    )


def test_id_number_fallback() -> None:
    boxes = [
        create_box(
            "CĂN CƯỚC CÔNG DÂN",
            300,
            200,
            800,
            250,
        ),
        create_box(
            "001095014159",
            400,
            300,
            780,
            350,
            0.99,
        ),
    ]

    parser = CCCDLayoutParser(boxes)

    result = parser.parse()

    assert result.id_number == (
        "001095014159"
    )


def test_date_cleaner() -> None:
    assert (
        CCCDLayoutParser.clean_date(
            "24 - 03 - 1995"
        )
        == "24/03/1995"
    )

    assert (
        CCCDLayoutParser.clean_date(
            "24031995"
        )
        == "24/03/1995"
    )


def test_id_cleaner() -> None:
    assert (
        CCCDLayoutParser.clean_id_number(
            "OO1O95O14159"
        )
        == "001095014159"
    )


def main() -> None:
    tests = (
        test_parse_all_fields,
        test_parse_to_dict,
        test_parse_with_debug,
        test_id_number_fallback,
        test_date_cleaner,
        test_id_cleaner,
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
            "Có test Layout Parser chưa đạt"
        )


if __name__ == "__main__":
    main()