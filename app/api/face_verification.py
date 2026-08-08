from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.exceptions import AppException, BadRequestException
from app.core.logger import logger
from app.modules.face_verification.errors import FaceVerificationError
from app.schemas.face_verification import (
    FaceVerificationErrorResponse,
    FaceVerificationResponse,
)
from app.services.face_verification_pipeline import face_verification_pipeline


router = APIRouter(
    prefix="/api/face-verification",
    tags=["Face Verification"],
)

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_upload_file(upload_file: UploadFile, field_name: str) -> None:
    if not upload_file.filename:
        raise BadRequestException(
            f"{field_name} chưa có tên file.",
            data={"errorCode": "MISSING_FILE_NAME", "field": field_name},
        )

    extension = Path(upload_file.filename).suffix.lower()
    content_type = (upload_file.content_type or "").lower()
    if (
        content_type not in ALLOWED_CONTENT_TYPES
        and extension not in ALLOWED_EXTENSIONS
    ):
        raise AppException(
            message=f"{field_name} phải là ảnh JPG, JPEG hoặc PNG.",
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            data={
                "errorCode": "UNSUPPORTED_IMAGE_TYPE",
                "field": field_name,
                "contentType": content_type or None,
                "extension": extension or None,
            },
        )


@router.post(
    "/verify",
    response_model=FaceVerificationResponse,
    response_model_exclude_none=True,
    summary="Đối chiếu khuôn mặt CCCD với ảnh webcam",
    description=(
        "Nhận ảnh mặt trước CCCD và một ảnh selfie chụp từ webcam, "
        "phát hiện đúng một khuôn mặt, kiểm tra chất lượng ảnh, sinh "
        "embedding ArcFace 512 chiều và so khớp bằng cosine similarity."
    ),
    responses={
        400: {
            "model": FaceVerificationErrorResponse,
            "description": "File đầu vào không hợp lệ.",
        },
        415: {
            "model": FaceVerificationErrorResponse,
            "description": "Định dạng ảnh không được hỗ trợ.",
        },
        422: {
            "model": FaceVerificationErrorResponse,
            "description": "Không tìm thấy khuôn mặt hoặc chất lượng ảnh không đạt.",
        },
        503: {
            "model": FaceVerificationErrorResponse,
            "description": "Model InsightFace chưa sẵn sàng.",
        },
        500: {
            "model": FaceVerificationErrorResponse,
            "description": "Lỗi nội bộ trong quá trình xác minh khuôn mặt.",
        },
    },
)
async def verify_face(
    card_image: Annotated[
        UploadFile,
        File(
            description=(
                "Ảnh mặt trước CCCD có chứa chân dung, định dạng JPG/JPEG/PNG."
            )
        ),
    ],
    webcam_image: Annotated[
        UploadFile,
        File(
            description=(
                "Ảnh selfie từ webcam: một người, nhìn thẳng, đủ sáng và rõ nét."
            )
        ),
    ],
) -> FaceVerificationResponse:
    try:
        validate_upload_file(card_image, "card_image")
        validate_upload_file(webcam_image, "webcam_image")

        read_limit = settings.max_upload_size_mb * 1024 * 1024 + 1
        card_image_bytes = await card_image.read(read_limit)
        webcam_image_bytes = await webcam_image.read(read_limit)

        pipeline_output = await run_in_threadpool(
            face_verification_pipeline.process,
            card_image_bytes,
            webcam_image_bytes,
        )
        return FaceVerificationResponse(**pipeline_output.to_dict())

    except AppException:
        raise
    except FaceVerificationError as exc:
        logger.warning(
            "Face verification rejected | code=%s | message=%s",
            exc.error_code,
            str(exc),
        )
        raise AppException(
            message=str(exc),
            status_code=exc.status_code,
            data=exc.to_data(),
        ) from exc
    except (ValueError, TypeError) as exc:
        logger.warning("Invalid face verification request: %s", exc)
        raise BadRequestException(
            str(exc),
            data={"errorCode": "INVALID_FACE_INPUT"},
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected face verification error")
        raise AppException(
            message="Đã xảy ra lỗi trong quá trình xác minh khuôn mặt.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            data={"errorCode": "FACE_VERIFICATION_INTERNAL_ERROR"},
        ) from exc
    finally:
        await card_image.close()
        await webcam_image.close()
