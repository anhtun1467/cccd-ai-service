from __future__ import annotations

from typing import Annotated

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
    status,
)
from starlette.concurrency import run_in_threadpool

from app.core.logger import logger
from app.schemas.face_verification import (
    FaceVerificationErrorResponse,
    FaceVerificationResponse,
)
from app.services.face_verification_pipeline import (
    face_verification_pipeline,
)


router = APIRouter(
    prefix="/api/face-verification",
    tags=["Face Verification"],
)


ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
}


def validate_upload_file(
    upload_file: UploadFile,
    field_name: str,
) -> None:
    """
    Kiểm tra định dạng file upload.
    """

    if not upload_file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} chưa có tên file.",
        )

    if upload_file.content_type not in ALLOWED_CONTENT_TYPES:
        allowed_types = ", ".join(
            sorted(ALLOWED_CONTENT_TYPES)
        )

        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"{field_name} không đúng định dạng ảnh. "
                f"Định dạng hỗ trợ: {allowed_types}."
            ),
        )


@router.post(
    "/verify",
    response_model=FaceVerificationResponse,
    summary="Đối chiếu khuôn mặt CCCD với ảnh webcam",
    description=(
        "Nhận ảnh mặt trước CCCD và ảnh khuôn mặt chụp "
        "từ webcam, sau đó thực hiện đối chiếu khuôn mặt 1:1."
    ),
    responses={
        400: {
            "model": FaceVerificationErrorResponse,
            "description": "File đầu vào không hợp lệ.",
        },
        415: {
            "model": FaceVerificationErrorResponse,
            "description": "Định dạng file không được hỗ trợ.",
        },
        422: {
            "model": FaceVerificationErrorResponse,
            "description": "Không thể phát hiện hoặc xác minh khuôn mặt.",
        },
        500: {
            "model": FaceVerificationErrorResponse,
            "description": "Lỗi nội bộ của hệ thống.",
        },
    },
)
async def verify_face(
    card_image: Annotated[
        UploadFile,
        File(
            description="Ảnh mặt trước CCCD dạng JPG, JPEG hoặc PNG."
        ),
    ],
    webcam_image: Annotated[
        UploadFile,
        File(
            description="Ảnh khuôn mặt chụp từ webcam."
        ),
    ],
) -> FaceVerificationResponse:
    """
    Đối chiếu ảnh chân dung trên CCCD với ảnh webcam.
    """

    validate_upload_file(
        card_image,
        field_name="card_image",
    )

    validate_upload_file(
        webcam_image,
        field_name="webcam_image",
    )

    try:
        card_image_bytes = await card_image.read()
        webcam_image_bytes = await webcam_image.read()

        # Pipeline xử lý AI là tác vụ CPU-bound.
        # Chạy trong threadpool để không chặn event loop FastAPI.
        pipeline_output = await run_in_threadpool(
            face_verification_pipeline.process,
            card_image_bytes,
            webcam_image_bytes,
        )

        return FaceVerificationResponse(
            **pipeline_output.to_dict()
        )

    except HTTPException:
        raise

    except (ValueError, TypeError) as exc:
        logger.warning(
            "Invalid face verification request: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        logger.warning(
            "Face verification failed: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Unexpected face verification error"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Đã xảy ra lỗi trong quá trình xác minh "
                "khuôn mặt."
            ),
        ) from exc

    finally:
        await card_image.close()
        await webcam_image.close()
