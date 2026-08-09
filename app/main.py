from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

from app.api import (
    face_verification,
    health,
    ocr,
)
from app.core.config import settings
from app.core.exception_handlers import (
    register_exception_handlers,
)
from app.core.logger import logger
from app.utils.file_utils import ensure_storage_dirs


def create_app() -> FastAPI:
    """
    Khởi tạo và cấu hình ứng dụng FastAPI.
    """

    ensure_storage_dirs()

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        description=(
            "AI Service xử lý CCCD gồm Card Detection, OCR và "
            "Face Verification.\n\n"
            "## Camera / tải ảnh selfie\n\n"
            "**[MỞ GIAO DIỆN CAMERA TRỰC TIẾP]"
            "(/face-verification)**\n\n"
            "Trang camera sẽ tự lấy `face_session.session_id` từ kết quả "
            "OCR; không cần nhập `session_id` hoặc gửi lại CCCD bằng tay."
        ),
        openapi_tags=[
            {
                "name": "Face Verification",
                "description": (
                    "Đối chiếu khuôn mặt 1:1. "
                    "**[Mở camera trực tiếp](/face-verification)** để dùng "
                    "webcam hoặc tải ảnh selfie lên."
                ),
            }
        ],
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(health.router)
    application.include_router(ocr.router)
    application.include_router(
        face_verification.router
    )

    face_ui_path = (
        Path(__file__).resolve().parent
        / "web"
        / "face_verification.html"
    )

    @application.get(
        "/",
        include_in_schema=False,
        response_class=RedirectResponse,
    )
    async def home() -> RedirectResponse:
        """Mở thẳng giao diện OCR -> camera/upload -> Face 1:1."""

        return RedirectResponse(
            url="/face-verification",
            status_code=307,
        )

    @application.get(
        "/camera",
        include_in_schema=False,
        response_class=RedirectResponse,
    )
    async def camera_shortcut() -> RedirectResponse:
        """Đường dẫn ngắn, dễ nhớ để mở giao diện camera."""

        return RedirectResponse(
            url="/face-verification",
            status_code=307,
        )

    @application.get(
        "/face-verification",
        include_in_schema=False,
        response_class=FileResponse,
    )
    async def face_verification_ui() -> FileResponse:
        """Trang thử luồng OCR -> camera/upload -> Face 1:1."""

        return FileResponse(
            face_ui_path,
            media_type="text/html; charset=utf-8",
        )

    register_exception_handlers(application)

    logger.info(
        "CCCD AI Service started successfully"
    )

    return application


app = create_app()
