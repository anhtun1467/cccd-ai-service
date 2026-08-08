from app.modules.ocr.validator import CCCDValidator


def _complete_data() -> dict[str, str | None]:
    return {
        "idNumber": "001304024809",
        "fullName": "HỒ NGỌC HÀ LINH",
        "dateOfBirth": "19/07/2004",
        "gender": "Nữ",
        "nationality": "Việt Nam",
        "placeOfOrigin": "Thống Nhất, Thường Tín, Hà Nội",
        "placeOfResidence": "Tổ 56 mới, Tương Mai, Hoàng Mai, Hà Nội",
        "dateOfExpiry": "19/07/2029",
    }


def test_complete_cccd_result_is_valid() -> None:
    result = CCCDValidator().validate(_complete_data())

    assert result["isValid"] is True
    assert all(result["fieldValidity"].values())


def test_missing_bottom_fields_cannot_be_reported_as_fully_valid() -> None:
    data = _complete_data()
    data["placeOfResidence"] = "Thôn 4B"
    data["dateOfExpiry"] = None

    result = CCCDValidator().validate(data)

    assert result["isValid"] is False
    assert result["fieldValidity"]["placeOfResidence"] is False
    assert result["fieldValidity"]["dateOfExpiry"] is False


def test_address_contaminated_by_expiry_noise_is_invalid() -> None:
    data = _complete_data()
    data["placeOfResidence"] = (
        "11 Ngách 35/72, giá 49424103/2035 Nguyễn Trãi, Hà Nội"
    )

    result = CCCDValidator().validate(data)

    assert result["isValid"] is False
    assert result["fieldValidity"]["placeOfResidence"] is False
