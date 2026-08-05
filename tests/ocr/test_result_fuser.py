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

    assert result["gender"] == "Nữ"
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
        "Lê Lợi, Thành phố Bắc Giang, Bắc Giang"
    )
    assert result["placeOfResidence"] == (
        "Phố Vôi, Thị trấn Vôi, Lạng Giang, Bắc Giang"
    )
    assert sources["placeOfOrigin"] == "RAW_TEXT_RECOVERY"
    assert sources["placeOfResidence"] == "RAW_TEXT_RECOVERY"


def test_residence_continues_after_expiry_line():
    result, sources = fuse_ocr_data(
        full_card_data={
            "placeOfResidence": "Place of residence TDP Dau Lang",
            "dateOfExpiry": "12/09/2030",
        },
        field_data={},
        raw_text=[
            "Noi thuong tru Place of residence TDP Dau Lang",
            "Cogia tri den 12/09/2030",
            "Thi tran Thanh Lang, Binh Xuyen; Vinh Phuc",
            "Date of Expiry",
        ],
    )

    assert result["placeOfResidence"] == (
        "TDP Dau Lang, Thị trấn Thanh Lang, Binh Xuyen, Vinh Phuc"
    )
    assert result["dateOfExpiry"] == "12/09/2030"
    assert sources["placeOfResidence"] == "RAW_TEXT_RECOVERY"


def test_fuses_female_and_addresses_from_noisy_real_ocr_sample():
    result, sources = fuse_ocr_data(
        full_card_data={
            "gender": None,
            "placeOfOrigin": (
                "Place of orgin, Hai Chau; Hai Hau, Nam Dinh"
            ),
            "placeOfResidence": (
                "Place of residence Phu Van Nam, "
                "Co Igia E (triden: 01/09/2024, "
                "Hai Chau; Hai Hau; Nam Dinh"
            ),
        },
        field_data={
            "gender": "Nß╗»",
            "placeOfOrigin": "Hải Châu, Hải Hậu, Nam Định",
            "placeOfResidence": (
                "Phú Vân Nam, Hải Châu, Hải Hậu, Nam Định"
            ),
        },
        raw_text=[
            "Gioi tinh / Sex: NG",
            'Que quan Place of orgin"',
            "Hai Chau; Hai Hau, Nam Dinh",
            "Noi thuong tru Place of residence Phu Van Nam",
            "Co Igia E (triden: 01/09/2024",
            "Hai Chau; Hai Hau; Nam Dinh",
            "Date of Expiry",
        ],
    )

    assert result["gender"] == "Nữ"
    assert sources["gender"] == "FIELD_OCR"
    assert result["placeOfOrigin"] == (
        "Hải Châu, Hải Hậu, Nam Định"
    )
    assert result["placeOfResidence"] == (
        "Phú Vân Nam, Hải Châu, Hải Hậu, Nam Định"
    )
    assert sources["placeOfOrigin"] == "RAW_TEXT_RECOVERY"
    assert sources["placeOfResidence"] == "RAW_TEXT_RECOVERY"


def test_address_recovery_preserves_vietnamese_accents():
    result, sources = fuse_ocr_data(
        full_card_data={},
        field_data={},
        raw_text=[
            "Quê quán / Place of origin:",
            "Hải Châu, Hải Hậu, Nam Định",
            "Nơi thường trú / Place of residence: Phú Vân Nam",
            "Hải Châu, Hải Hậu, Nam Định",
        ],
    )

    assert result["placeOfOrigin"] == (
        "Hải Châu, Hải Hậu, Nam Định"
    )
    assert result["placeOfResidence"] == (
        "Phú Vân Nam, Hải Châu, Hải Hậu, Nam Định"
    )
    assert sources["placeOfOrigin"] == "RAW_TEXT_RECOVERY"
    assert sources["placeOfResidence"] == "RAW_TEXT_RECOVERY"


def test_noisy_labels_and_mixed_columns_from_real_card_sample():
    result, sources = fuse_ocr_data(
        full_card_data={
            "gender": "Nữ",
            "placeOfOrigin": (
                "Place ol ongin, Bách Thuân, Vù Thư, Thái Binh, "
                "Dalctokoro Cogla àoen 1/01/2028 Ea Hiao, "
                "Ea Hleo, Đák Lák Noi thưong tú / "
                "Place of 'residonceThôn 48"
            ),
        },
        field_data={
            "placeOfOrigin": "Bach Thuan, Vú Thu, Thai Binh",
            "placeOfResidence": (
                "Inhon 4B, Ea, Ea Hleo, Đák Lắk, Hlao"
            ),
            "dateOfExpiry": "01/01/2028",
        },
        raw_text=[
            "Sex Nữ",
            "Que quan / Place ol ongin:",
            "Bách Thuân, Vù Thư, Thái Binh",
            (
                "Dalctokoro Cogla àoen 1/01/2028 "
                "Ea Hiao, Ea Hleo, Đák Lák "
                "Noi thưong tú / Place of 'residonceThôn 48"
            ),
        ],
    )

    assert result["gender"] == "Nữ"
    assert result["placeOfOrigin"] == (
        "Bách Thuận, Vũ Thư, Thái Bình"
    )
    assert result["placeOfResidence"] == (
        "Thôn 4B, Ea Hiao, Ea H'Leo, Đắk Lắk"
    )
    assert result["dateOfExpiry"] == "01/01/2028"
    assert sources["placeOfOrigin"] == "RAW_TEXT_RECOVERY"
    assert sources["placeOfResidence"] == "RAW_TEXT_RECOVERY"


def test_real_card_024_recovers_clean_addresses_name_and_expiry():
    result, sources = fuse_ocr_data(
        full_card_data={
            "idNumber": "024199006144",
            "fullName": "TRƯƠNG PHƯƠNG HUYỀN",
            "dateOfBirth": "24/02/1999",
            "gender": "Nữ",
            "nationality": "Nationallty Viêt Nam",
            "placeOfOrigin": (
                "Place of origin, Lê Lợi, Thành phổ Bẳc Giang; "
                "Bác Giang, No thửởng trú"
            ),
            "placeOfResidence": (
                "Phố Vôi, Co gia ưn đen:, Duto of erpiry "
                "24/02/2024 Thị trấn Vôi, Lang Giang, Bắc Giang"
            ),
            "dateOfExpiry": None,
        },
        field_data={
            "dateOfExpiry": None,
        },
        raw_text=[
            "56/No: 024199006144",
            "Ho va ten / Full name",
            "TRƯƠNG PHƯƠNG HUYỀN",
            "Ngay sinh Date of birth: 24/02/1999",
            "Quoc tich Nationallty: Viêt Nam",
            "Sex: Nữ",
            "Giol unh",
            "Que quan Place of origin ;",
            "Lê Lợi, Thành phổ Bẳc Giang; Bác Giang",
            "No thửởng trú",
            "Place of residence Phố Vôi",
            "Co gia ưn đen:",
            (
                "Duto of erpiry 24/02/2024 "
                "Thị trấn Vôi, Lang Giang, Bắc Giang"
            ),
        ],
    )

    assert result["fullName"] == "TRƯƠNG PHƯƠNG HUYỀN"
    assert result["placeOfOrigin"] == (
        "Lê Lợi, Thành phố Bắc Giang, Bắc Giang"
    )
    assert result["placeOfResidence"] == (
        "Phố Vôi, Thị trấn Vôi, Lạng Giang, Bắc Giang"
    )
    assert result["dateOfExpiry"] == "24/02/2024"
    assert sources["placeOfOrigin"] == "RAW_TEXT_RECOVERY"
    assert sources["placeOfResidence"] == "RAW_TEXT_RECOVERY"
    assert sources["dateOfExpiry"] == "RAW_TEXT_RECOVERY"


def test_real_card_046_preserves_and_reconciles_name_accents():
    result, sources = fuse_ocr_data(
        full_card_data={
            "idNumber": "046304000912",
            "fullName": "SÓ NO",
            "dateOfBirth": "06/12/2004",
            "gender": "Nữ",
            "nationality": "Việt Nam",
        },
        field_data={
            "fullName": "HÔ TÙNG CHI",
            "placeOfResidence": (
                "46, /, Phú Tuyên, Bình Thành, Hương, "
                "Thừa Thiên Huế, Trà"
            ),
            "dateOfExpiry": "06/12/2029",
        },
        raw_text=[
            "Só / No. 046304000912",
            "Ho va ten / Full name:",
            "HỔ TÙNG CHI",
            "Ngay sinh / Date of birth: 06/12/2004",
            "Gioi tinh / Sex: Nữ",
            "Quoc tich Nationality: Việt Nam",
            "Que quan Place of origin :",
            "Vinh Hà, Phú Vang, Thừa Thiên Huế",
            "Noi thuong tru / Place of residence : Phú Tuyên",
            "Cỏ giá trịđến: 06/12/2029",
            "Binh Thành, Hương Trà, Thừa Thiên Huế",
            "Date of Expiry",
        ],
    )

    assert result["fullName"] == "HỒ TÙNG CHI"
    assert result["placeOfOrigin"] == (
        "Vinh Hà, Phú Vang, Thừa Thiên Huế"
    )
    assert result["placeOfResidence"] == (
        "Phú Tuyên, Bình Thành, Hương Trà, Thừa Thiên Huế"
    )
    assert result["dateOfExpiry"] == "06/12/2029"
    assert sources["fullName"] == "RAW_TEXT_RECOVERY"


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


def test_expiry_does_not_use_birth_date_or_co_gia_dinh_address():
    result, _ = fuse_ocr_data(
        full_card_data={},
        field_data={},
        raw_text=[
            "Ngay sinh / Date of birth: 11/01/2003",
            "Noi thuong tru / Place of residence:",
            "Cô Gia Định, Phường 2, Thành phố Hồ Chí Minh",
        ],
    )

    assert result["dateOfExpiry"] is None
    assert result["placeOfResidence"] == (
        "Cô Gia Định, Phường 2, Thành phố Hồ Chí Minh"
    )

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

    assert normalize_gender("Nu") == "Nữ"
    assert normalize_gender("Nữ") == "Nữ"
    assert normalize_gender("Nß╗»") == "Nữ"
    assert normalize_gender("Female") == "Nữ"


def test_recover_female_before_nationality():
    from app.modules.ocr.result_fuser import recover_gender

    raw_text = [
        "Gioi tinh / Sex: Nu "
        "Quoc tich / Nationality: Viet Nam"
    ]

    assert recover_gender(raw_text) == "Nữ"


def test_recover_male_before_nationality():
    from app.modules.ocr.result_fuser import recover_gender

    raw_text = [
        "Gioi tinh / Sex: Nam "
        "Quoc tich / Nationality: Viet Nam"
    ]

    assert recover_gender(raw_text) == "Nam"


def test_restores_missing_diacritics_for_female_name_from_real_cards():
    result, sources = fuse_ocr_data(
        full_card_data={
            "fullName": "ĐĂNG THI MAY",
            "gender": "Nữ",
        },
        field_data={},
        raw_text=[
            "Ho va ten / Full name:",
            "ĐĂNG THI MAY",
            "Gioi tinh / Sex: Nữ",
        ],
    )

    assert result["fullName"] == "ĐẶNG THỊ MÂY"
    assert sources["fullName"] == "FULL_CARD_OCR"


def test_restores_common_surname_without_changing_unambiguous_words():
    result, _ = fuse_ocr_data(
        full_card_data={},
        field_data={},
        raw_text=[
            "Ho va ten / Full name:",
            "PHAM THỊ NGOAN",
            "Gioi tinh / Sex: Nữ",
        ],
    )

    assert result["fullName"] == "PHẠM THỊ NGOAN"


def test_address_accent_recovery_does_not_change_thanh_hoa_to_thanh_hoa():
    result, _ = fuse_ocr_data(
        full_card_data={},
        field_data={
            "placeOfResidence": (
                "18 Nguyên Hồng, Tân Sơn, Thành phố Thanh Hóa, Thanh Hóa"
            ),
        },
        raw_text=[
            "Noi thuong tru / Place of residence: 18 Nguyen Hong",
            "Tan Son, Thanh pho Thanh Hoa, Thanh Hoa",
        ],
    )

    assert result["placeOfResidence"] == (
        "18 Nguyên Hồng, Tân Sơn, Thành phố Thanh Hóa, Thanh Hóa"
    )


def test_removes_onigin_and_dato_atorpiry_noise_from_real_card():
    result, _ = fuse_ocr_data(
        full_card_data={},
        field_data={
            "placeOfOrigin": "Hải Châu, Hải Hậu, Nam Định",
            "placeOfResidence": (
                "Phú Vân Nam, Hải Châu, Hải Hậu, Nam Định"
            ),
        },
        raw_text=[
            "Que quan Place of onigin:",
            "Hải Châu, Hải Hậu, Nam Đinh",
            "Noi thuong tru Place of residence",
            "Phú Văn Nam",
            "Co Qia triden 01/09/2024",
            "Hải Châu, Hải Hậu, Nam Định",
            "Dato atorpiry",
        ],
    )

    assert result["placeOfOrigin"] == "Hải Châu, Hải Hậu, Nam Định"
    assert result["placeOfResidence"] == (
        "Phú Vân Nam, Hải Châu, Hải Hậu, Nam Định"
    )
