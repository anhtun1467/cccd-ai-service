from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.exceptions import AppException
from app.core.logger import logger


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        logger.warning(
            "AppException | path=%s | message=%s",
            request.url.path,
            exc.message,
        )

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "message": exc.message,
                "data": exc.data,
            },
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception(
            "Unhandled Exception | path=%s | error=%s",
            request.url.path,
            str(exc),
        )

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Lỗi hệ thống không xác định",
                "data": None,
            },
        )