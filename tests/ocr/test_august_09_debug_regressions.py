from app.modules.ocr.result_fuser import fuse_ocr_data
from app.modules.ocr.validator import CCCDValidator


def _candidate(*values: str) -> dict:
    return {
        "ocrCandidates": [
            {
                "variant": f"test_{index}",
                "value": value,
                "joinedText": value,
                "rawText": [value],
                "normalizedText": [value],
            }
            for index, value in enumerate(values)
        ]
    }


def test_blurry_00fc_uses_all_field_variants() -> None:
    result, _ = fuse_ocr_data(
        full_card_data={},
        field_data={
            "idNumber": "027191001864",
            "fullName": "DUONG THI HƯƠNG HIÊP",
            "nationality": "Việt Nam",
            "placeOfOrigin": "grasKhê Tni sà Tu Sợn Bác Ninh",
            "placeOfResidence": (
                "aurhduoguouo, Đinn Bảng Thỉxẳ Tử Son "
                "Roinuuryu, Fnlo crusdtyre ' hint an Lang"
            ),
        },
        raw_text=[
            "30 40 027191001864",
            "DUONG THI HƯƠNG HIfP",
            "Chau Khe Inl wa Tu Son Buc Mlinh",
            "Đinh Bang Thixả Tử Son Bac Minh",
        ],
        field_results={
            "fullName": _candidate(
                "DUONG THI HƯƠNG HIÊP",
                "DUONG THI HƯƠNG HIÊP",
            ),
            "dateOfBirth": _candidate("17/101001"),
            "placeOfOrigin": _candidate(
                "grasKhê Tni sà Tu Sợn Bác Ninh",
            ),
            "placeOfResidence": _candidate(
                "Rolnory uuHca crrgstk Thint Lang, "
                "Đinh Bảng Thịxả Tử Son Bac Minh",
            ),
            "dateOfExpiry": _candidate("1710201", "17/0201"),
        },
    )

    assert result == {
        "idNumber": "027191001864",
        "fullName": "DƯƠNG THỊ HƯƠNG HIỆP",
        "dateOfBirth": "17/10/1991",
        "gender": "Nữ",
        "nationality": "Việt Nam",
        "placeOfOrigin": "Châu Khê, Thị xã Từ Sơn, Bắc Ninh",
        "placeOfResidence": (
            "Thịnh Lang, Đình Bảng, Thị xã Từ Sơn, Bắc Ninh"
        ),
        "dateOfExpiry": "17/10/2031",
    }


def test_54b2_drops_birth_label_tail_from_name_and_recovers_bottom() -> None:
    result, _ = fuse_ocr_data(
        full_card_data={
            "idNumber": "034187008200",
            "nationality": "Việt Nam",
        },
        field_data={
            "fullName": "BỦL THI HONG INH DAETUNN",
            "dateOfBirth": "16/02/1987",
            "placeOfOrigin": "Sam Binh, Ken Xưang Thài Binh",
            "placeOfResidence": (
                "Picn d rrsiDior Cao Mai Đoàl, Quang Trung, "
                "Kiên Xương, Thal Binh"
            ),
        },
        raw_text=[
            "S6/m 034187008200",
            "BỦl THI HỎNG",
            "Nqay cinh / Deethnthr 16/02/1987",
        ],
        field_results={
            "fullName": _candidate(
                "BỦL THI HONG EUBNR AN",
                "BỦL THI HONG",
                "BỦL THI HONG INH DAETUNN",
                "BỦL THI HONG",
            ),
            "dateOfExpiry": _candidate(
                "509 3 100272027",
                "529 86 100212027",
            ),
        },
    )

    assert result["fullName"] == "BÙI THỊ HỒNG"
    assert result["placeOfOrigin"] == "Nam Bình, Kiến Xương, Thái Bình"
    assert result["placeOfResidence"] == (
        "Cao Mai Đoài, Quang Trung, Kiến Xương, Thái Bình"
    )
    assert result["dateOfExpiry"] == "16/02/2027"


def test_701a_removes_corrupt_expiry_blob_from_residence() -> None:
    result, sources = fuse_ocr_data(
        full_card_data={
            "fullName": "NGUYỄN HOÀNG NAM",
            "dateOfBirth": "24/03/1995",
            "gender": "Nam",
            "nationality": "Nam Việt Nam",
            "placeOfOrigin": "Bạch Đằng, Hai Bà Trưng, Hà Nội",
            "placeOfResidence": (
                "11 Ngách 35/72, giá 49424103/2035 Nguyễn Trãi, "
                "Nhân Chính, Thanhgacân, Hà Nc"
            ),
        },
        field_data={"idNumber": "001095014159"},
        raw_text=[
            "Số / No: 001095014159",
            "Nơi thường trú / Place of residence: 1 1 Ngách 35/72",
            (
                "giá 49424103/2035 Nguyễn Trãi, Nhân Chính, "
                "Thanhgacân, Hà Nc"
            ),
        ],
        field_results={
            "dateOfExpiry": _candidate("8096 4.96724/03/2035"),
        },
    )

    assert result["placeOfResidence"] == (
        "11 Ngách 35/72, Nguyễn Trãi, Nhân Chính, Thanh Xuân, Hà Nội"
    )
    assert result["dateOfExpiry"] == "24/03/2035"
    assert sources["placeOfResidence"] in {
        "RAW_TEXT_RECOVERY",
        "FULL_CARD_OCR",
    }


def test_1472_reorders_house_number_and_street_from_crossed_lines() -> None:
    result, _ = fuse_ocr_data(
        full_card_data={
            "idNumber": "038203000066",
            "fullName": "LƯU HUY MINH",
            "dateOfBirth": "14/03/2003",
            "gender": "Nam",
            "nationality": "Việt Nam",
            "placeOfOrigin": "Đông Sơn, Thanh Hóa",
        },
        field_data={
            "placeOfResidence": (
                "18 Tân Sơn, Thành phố Thanh Hóà, Aoguyêanồnga Hồng"
            ),
            "dateOfExpiry": "14/03/2028",
        },
        raw_text=[],
    )

    assert result["placeOfResidence"] == (
        "18 Nguyên Hồng, Tân Sơn, Thành phố Thanh Hóa, Thanh Hóa"
    )


def test_9240_repairs_extra_digit_in_expiry_using_birth_milestone() -> None:
    result, _ = fuse_ocr_data(
        full_card_data={
            "idNumber": "092195009838",
            "fullName": "TRƯƠNG THỊ YẾN NHI",
            "dateOfBirth": "05/01/1995",
            "gender": "Nữ",
            "nationality": "Việt Nam",
            "placeOfOrigin": "Thới An Đông, Bình Thủy, Cần Thơ",
            "placeOfResidence": (
                "KV Thới Thuận, Phước Thới, Ô Môn, Cần Thơ"
            ),
        },
        field_data={},
        raw_text=["Có giá trị đến: 05/0772035"],
        field_results={
            "dateOfExpiry": _candidate(
                "909.18 05/0172035 1",
                "209818 05/0172035 1",
            ),
        },
    )

    assert result["dateOfExpiry"] == "05/01/2035"
    assert CCCDValidator().validate(result)["isValid"] is True


def test_cropped_bottom_is_not_fabricated_without_expiry_evidence() -> None:
    result, _ = fuse_ocr_data(
        full_card_data={
            "idNumber": "066303000485",
            "fullName": "ĐẶNG THỊ MÂY",
            "dateOfBirth": "11/01/2003",
            "gender": "Nữ",
            "nationality": "Việt Nam",
            "placeOfOrigin": "Bách Thuận, Vũ Thư, Thái Bình",
            "placeOfResidence": "Thôn 4B",
        },
        field_data={},
        raw_text=[],
        field_results={"dateOfExpiry": _candidate("7", "3")},
    )

    assert result["dateOfExpiry"] is None
    validation = CCCDValidator().validate(result)
    assert validation["isValid"] is False
    assert validation["fieldValidity"]["dateOfExpiry"] is False
