from __future__ import annotations

from app.modules.qr.cccd_qr_parser import CCCDQRParser


def test_parse_seven_field_cccd_qr_without_mapping_issue_to_expiry() -> None:
    payload = (
        "042096015766||ĐINH XUÂN HOÀNG|24111996|Nam|"
        "Lâm Trung Thủy, Đức Thọ, Hà Tĩnh|01122021\r\n"
    )

    result = CCCDQRParser().parse(payload)

    assert result["success"] is True
    assert result["format"] == "CCCD_QR_7_FIELDS"
    assert result["structuredData"] == {
        "idNumber": "042096015766",
        "fullName": "ĐINH XUÂN HOÀNG",
        "dateOfBirth": "24/11/1996",
        "gender": "Nam",
        "placeOfResidence": "Lâm Trung Thủy, Đức Thọ, Hà Tĩnh",
    }
    assert "dateOfExpiry" not in result["structuredData"]
    assert "placeOfOrigin" not in result["structuredData"]
    assert "nationality" not in result["structuredData"]
    assert result["auxiliaryData"]["hasDateOfIssue"] is True


def test_parse_extended_qr_keeps_extra_fields_out_of_cccd_data() -> None:
    payload = (
        "001200000001|123456789|NGUYỄN VĂN AN|01012000|Nữ|"
        "Phường A, Quận B, Hà Nội|02022024|cancelled-id|"
        "father-name|mother-name|reserved"
    )

    result = CCCDQRParser().parse(payload)

    assert result["success"] is True
    assert result["format"] == "CAN_CUOC_QR_EXTENDED"
    assert result["fieldCount"] == 11
    assert result["structuredData"]["gender"] == "Nữ"
    assert result["auxiliaryData"]["additionalFieldCount"] == 4
    assert result["auxiliaryData"]["hasOldDocumentNumber"] is True


def test_random_or_malformed_qr_is_not_accepted_as_cccd() -> None:
    parser = CCCDQRParser()

    assert parser.parse("https://example.com/pay?id=123")["success"] is False
    invalid_id = parser.parse(
        "ABC||NGUYỄN VĂN AN|01012000|Nam|Phường A, Quận B|02022024"
    )
    assert invalid_id["success"] is False
    assert invalid_id["structuredData"] == {}
    assert "QR_ID_NUMBER_INVALID" in invalid_id["errors"]
    assert "idNumber" in invalid_id["missingRequiredFields"]
    assert any(
        item["code"] == "QR_ID_NUMBER_INVALID"
        and item["stage"] == "parse"
        for item in invalid_id["errorDetails"]
    )
