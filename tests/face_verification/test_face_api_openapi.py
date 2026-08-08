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

