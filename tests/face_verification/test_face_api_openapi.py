from __future__ import annotations

from fastapi import FastAPI

from app.api.face_verification import router


def test_face_verification_openapi_uses_valid_vietnamese() -> None:
    app = FastAPI()
    app.include_router(router)
    schema = app.openapi()
    operation = schema["paths"]["/api/face-verification/verify"]["post"]

    assert operation["summary"] == "Đối chiếu khuôn mặt CCCD với ảnh webcam"
    assert "ArcFace 512 chiều" in operation["description"]
    assert "?" not in operation["summary"]

    body_schema = operation["requestBody"]["content"][
        "multipart/form-data"
    ]["schema"]
    reference = body_schema["$ref"].split("/")[-1]
    properties = schema["components"]["schemas"][reference]["properties"]

    assert properties["card_image"]["description"].startswith("Ảnh mặt trước")
    assert properties["webcam_image"]["description"].startswith("Ảnh selfie")


def test_integrated_face_endpoint_does_not_request_cccd_again() -> None:
    app = FastAPI()
    app.include_router(router)
    schema = app.openapi()
    operation = schema["paths"][
        "/api/face-verification/verify-from-ocr"
    ]["post"]

    body_schema = operation["requestBody"]["content"][
        "multipart/form-data"
    ]["schema"]
    reference = body_schema["$ref"].split("/")[-1]
    properties = schema["components"]["schemas"][reference]["properties"]

    assert set(properties) == {
        "session_id",
        "selfie_image",
        "capture_source",
    }
    assert "card_image" not in properties
    assert "không phải gửi CCCD lần thứ hai" in operation["description"]
