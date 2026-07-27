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
    Ki?m tra d?nh d?ng file upload.
    """

    if not upload_file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field_name} chua có tên file.",
        )

    if upload_file.content_type not in ALLOWED_CONTENT_TYPES:
        allowed_types = ", ".join(
            sorted(ALLOWED_CONTENT_TYPES)
        )

        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"{field_name} không dúng d?nh d?ng ?nh. "
                f"Ð?nh d?ng h? tr?: {allowed_types}."
            ),
        )


@router.post(
    "/verify",
    response_model=FaceVerificationResponse,
    summary="Ð?i chi?u khuôn m?t CCCD v?i ?nh webcam",
    description=(
        "Nh?n ?nh m?t tru?c CCCD và ?nh khuôn m?t ch?p "
        "t? webcam, sau dó th?c hi?n d?i chi?u khuôn m?t 1:1."
    ),
    responses={
        400: {
            "model": FaceVerificationErrorResponse,
            "description": "File d?u vào không h?p l?.",
        },
        415: {
            "model": FaceVerificationErrorResponse,
            "description": "Ð?nh d?ng file không du?c h? tr?.",
        },
        422: {
            "model": FaceVerificationErrorResponse,
            "description": "Không th? phát hi?n ho?c xác minh khuôn m?t.",
        },
        500: {
            "model": FaceVerificationErrorResponse,
            "description": "L?i n?i b? c?a h? th?ng.",
        },
    },
)
async def verify_face(
    card_image: Annotated[
        UploadFile,
        File(
            description="?nh m?t tru?c CCCD d?ng JPG, JPEG ho?c PNG."
        ),
    ],
    webcam_image: Annotated[
        UploadFile,
        File(
            description="?nh khuôn m?t ch?p t? webcam."
        ),
    ],
) -> FaceVerificationResponse:
    """
    Ð?i chi?u ?nh chân dung trên CCCD v?i ?nh webcam.
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

        # Pipeline x? lý AI là tác v? CPU-bound.
        # Ch?y trong threadpool d? không ch?n event loop FastAPI.
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
                "Ðã x?y ra l?i trong quá trình xác minh "
                "khuôn m?t."
            ),
        ) from exc

    finally:
        await card_image.close()
        await webcam_image.close()

