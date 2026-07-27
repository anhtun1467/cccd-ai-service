from app.modules.ocr.result_fuser import (
    fuse_ocr_data,
    normalize_gender,
)


def test_gender_does_not_use_viet_nam_as_male():
    assert normalize_gender("Viet Nam") is None


def test_female_recovery_from_same_line():
    result, sources = fuse_ocr_data(
        full_card_data={
            "gender": "Nam",
            "nationality": "Viet Nam",
        },
        field_data={},
        raw_text=[
            "Gioi tinh / Sex: Nu Quoc tich / Nationality: Viet Nam",
        ],
    )

    assert result["gender"] == "Nu"
    assert sources["gender"] == "RAW_TEXT_RECOVERY"


def test_name_is_truncated_before_birth_label():
    result, _ = fuse_ocr_data(
        full_card_data={
            "fullName": "DANG THI MAY NGAY SINH DATE OF BIRTH",
        },
        field_data={},
        raw_text=[
            "Ho va ten / Full name:",
            "DANG THI MAY",
            "Ngay sinh / Date of birth: 11/01/2003",
        ],
    )

    assert result["fullName"] == "DANG THI MAY"


def test_origin_and_residence_from_raw_lines():
    result, sources = fuse_ocr_data(
        full_card_data={
            "placeOfOrigin": "Place of origin:",
            "placeOfResidence": "Place of residence:",
        },
        field_data={},
        raw_text=[
            "Que quan / Place of origin:",
            "Le Loi, Thanh pho Bac Giang, Bac Giang",
            "Noi thuong tru / Place of residence: Pho Voi",
            "Thi tran Voi, Lang Giang, Bac Giang",
        ],
    )

    assert result["placeOfOrigin"] == (
        "Le Loi, Thanh pho Bac Giang, Bac Giang"
    )
    assert result["placeOfResidence"] == (
        "Pho Voi, Thi tran Voi, Lang Giang, Bac Giang"
    )
    assert sources["placeOfOrigin"] == "RAW_TEXT_RECOVERY"
    assert sources["placeOfResidence"] == "RAW_TEXT_RECOVERY"


def test_expiry_is_recovered_from_expiry_label_only():
    result, _ = fuse_ocr_data(
        full_card_data={"dateOfExpiry": None},
        field_data={"dateOfExpiry": None},
        raw_text=[
            "Ngay sinh / Date of birth: 11/01/2003",
            "Co gia tri den: 11/01/2028",
        ],
    )

    assert result["dateOfExpiry"] == "11/01/2028"

def test_gender_parser_does_not_read_viet_nam_as_male():
    from app.modules.ocr.result_fuser import normalize_gender

    assert normalize_gender("Viet Nam") is None
    assert normalize_gender("Việt Nam") is None


def test_gender_parser_returns_nam():
    from app.modules.ocr.result_fuser import normalize_gender

    assert normalize_gender("Nam") == "Nam"
    assert normalize_gender("Male") == "Nam"


def test_gender_parser_returns_nu():
    from app.modules.ocr.result_fuser import normalize_gender

    assert normalize_gender("Nu") == "Nu"
    assert normalize_gender("Nữ") == "Nu"
    assert normalize_gender("Female") == "Nu"


def test_recover_female_before_nationality():
    from app.modules.ocr.result_fuser import recover_gender

    raw_text = [
        "Gioi tinh / Sex: Nu "
        "Quoc tich / Nationality: Viet Nam"
    ]

    assert recover_gender(raw_text) == "Nu"


def test_recover_male_before_nationality():
    from app.modules.ocr.result_fuser import recover_gender

    raw_text = [
        "Gioi tinh / Sex: Nam "
        "Quoc tich / Nationality: Viet Nam"
    ]

    assert recover_gender(raw_text) == "Nam"

