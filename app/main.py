from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
            "AI Service xử lý CCCD gồm Card Detection, "
            "OCR và Face Verification."
        ),
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

    register_exception_handlers(application)

    logger.info(
        "CCCD AI Service started successfully"
    )

    return application


app = create_app()
