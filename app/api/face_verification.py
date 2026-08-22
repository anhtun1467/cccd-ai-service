from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.exceptions import AppException, BadRequestException
from app.core.logger import logger
from app.modules.face_verification.errors import FaceVerificationError
from app.schemas.face_verification import (
    CaptureSource,
    FaceVerificationErrorResponse,
    FaceVerificationFromOcrResponse,
    FaceVerificationResponse,
    FaceSessionResponse,
    FaceMetrics,
)
from app.services.face_verification_pipeline import face_verification_pipeline
from app.services.face_session_store import (
    FaceSession,
    FaceSessionError,
    face_session_store,
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
    deprecated=True,
    description=(
        "Nhận ảnh mặt trước CCCD và một ảnh selfie chụp từ webcam, "
        "phát hiện đúng một khuôn mặt, kiểm tra chất lượng ảnh, sinh "
        "embedding ArcFace 512 chiều và so khớp bằng cosine similarity. "
        "Endpoint tương thích cũ; luồng mới nên dùng /verify-from-ocr để "
        "không phải gửi CCCD lần thứ hai."
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
        return FaceVerificationResponse(
        **pipeline_output.to_dict(),
        metrics=FaceMetrics()
    )

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


def _raise_session_error(exc: FaceSessionError) -> None:
    raise AppException(
        message=str(exc),
        status_code=exc.status_code,
        data=exc.to_data(),
    ) from exc


def _record_rejected_attempt(
    session: FaceSession | None,
    *,
    capture_source: CaptureSource,
    error_code: str,
) -> dict[str, object] | None:
    if session is None:
        return None
    updated = face_session_store.record_attempt(
        session.session_id,
        verification_status="error",
        capture_source=capture_source,
        error_code=error_code,
    )
    return updated.to_public_dict()


@router.post(
    "/verify-from-ocr",
    response_model=FaceVerificationFromOcrResponse,
    response_model_exclude_none=True,
    summary="Đối chiếu selfie bằng ảnh CCCD đã xử lý ở bước OCR",
    description=(
        "Chỉ nhận face session id do POST /ocr/cccd trả về và một ảnh "
        "selfie từ camera hoặc upload. Server tự lấy crop chân dung/ảnh "
        "CCCD đã làm phẳng của đúng phiên OCR; client không phải gửi CCCD "
        "lần thứ hai và không được truyền đường dẫn file."
    ),
    responses={
        400: {
            "model": FaceVerificationErrorResponse,
            "description": "Session id hoặc ảnh selfie không hợp lệ.",
        },
        404: {
            "model": FaceVerificationErrorResponse,
            "description": "Không tìm thấy phiên OCR.",
        },
        409: {
            "model": FaceVerificationErrorResponse,
            "description": "Phiên đã đối chiếu thành công.",
        },
        410: {
            "model": FaceVerificationErrorResponse,
            "description": "Phiên đã hết hạn hoặc ảnh OCR không còn tồn tại.",
        },
        415: {
            "model": FaceVerificationErrorResponse,
            "description": "Định dạng selfie không được hỗ trợ.",
        },
        422: {
            "model": FaceVerificationErrorResponse,
            "description": "Không tìm thấy mặt hoặc chất lượng selfie không đạt.",
        },
        429: {
            "model": FaceVerificationErrorResponse,
            "description": "Phiên đã hết số lần thử.",
        },
        503: {
            "model": FaceVerificationErrorResponse,
            "description": "Model InsightFace chưa sẵn sàng.",
        },
    },
)
async def verify_face_from_ocr(
    session_id: Annotated[
        str,
        Form(
            description=(
                "face_session.session_id nhận từ response POST /ocr/cccd."
            )
        ),
    ],
    selfie_image: Annotated[
        UploadFile,
        File(
            description=(
                "Ảnh selfie JPG/JPEG/PNG; đúng một người, nhìn thẳng và đủ sáng."
            )
        ),
    ],
    capture_source: Annotated[
        CaptureSource,
        Form(
            description=(
                "Nguồn ảnh phía giao diện để phục vụ audit UX; không phải "
                "bằng chứng liveness."
            )
        ),
    ] = "upload",
) -> FaceVerificationFromOcrResponse:
    session: FaceSession | None = None
    try:
        validate_upload_file(selfie_image, "selfie_image")
        read_limit = settings.max_upload_size_mb * 1024 * 1024 + 1
        selfie_image_bytes = await selfie_image.read(read_limit)

        with face_session_store.verification_lease(session_id) as leased_session:
            session = leased_session
            card_image_path = face_session_store.resolve_card_image_path(
                session
            )
            portrait_image_path = (
                face_session_store.resolve_portrait_image_path(session)
            )

            pipeline_output = await run_in_threadpool(
                face_verification_pipeline.process_from_ocr_paths,
                card_image_path=card_image_path,
                portrait_image_path=portrait_image_path,
                webcam_image_bytes=selfie_image_bytes,
            )
            updated_session = face_session_store.record_attempt(
                session.session_id,
                verification_status=pipeline_output.verification.status,
                capture_source=capture_source,
            )

        response_data = pipeline_output.to_dict()
        response_data.update(
            {
                "ocr_session_id": session.session_id,
                "capture_source": capture_source,
                "session": updated_session.to_public_dict(),
            }
        )
        return FaceVerificationFromOcrResponse(**response_data)

    except FaceSessionError as exc:
        logger.warning(
            "Face session rejected | code=%s | message=%s",
            exc.error_code,
            str(exc),
        )
        _raise_session_error(exc)
        raise AssertionError("unreachable")
    except AppException:
        raise
    except FaceVerificationError as exc:
        logger.warning(
            "Face verification from OCR rejected | code=%s | message=%s",
            exc.error_code,
            str(exc),
        )
        error_data = exc.to_data()
        if exc.status_code in {400, 422}:
            session_data = _record_rejected_attempt(
                session,
                capture_source=capture_source,
                error_code=exc.error_code,
            )
            if session_data is not None:
                error_data["session"] = session_data
        raise AppException(
            message=str(exc),
            status_code=exc.status_code,
            data=error_data,
        ) from exc
    except (ValueError, TypeError) as exc:
        logger.warning("Invalid selfie from OCR session: %s", exc)
        session_data = _record_rejected_attempt(
            session,
            capture_source=capture_source,
            error_code="INVALID_SELFIE_IMAGE",
        )
        data: dict[str, object] = {
            "errorCode": "INVALID_SELFIE_IMAGE",
        }
        if session_data is not None:
            data["session"] = session_data
        raise BadRequestException(str(exc), data=data) from exc
    except Exception as exc:
        logger.exception("Unexpected face verification from OCR error")
        raise AppException(
            message="Đã xảy ra lỗi trong quá trình xác minh khuôn mặt.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            data={"errorCode": "FACE_VERIFICATION_INTERNAL_ERROR"},
        ) from exc
    finally:
        await selfie_image.close()


@router.get(
    "/sessions/{session_id}",
    response_model=FaceSessionResponse,
    response_model_exclude_none=True,
    summary="Kiểm tra trạng thái phiên Face được tạo từ OCR",
)
async def get_face_session(session_id: str) -> FaceSessionResponse:
    try:
        session = face_session_store.get_public(session_id)
        return FaceSessionResponse(**session.to_public_dict())
    except FaceSessionError as exc:
        _raise_session_error(exc)
        raise AssertionError("unreachable")


@router.delete(
    "/sessions/{session_id}",
    response_model=FaceSessionResponse,
    response_model_exclude_none=True,
    summary="Hủy phiên Face được tạo từ OCR",
)
async def cancel_face_session(session_id: str) -> FaceSessionResponse:
    try:
        session = face_session_store.cancel(session_id)
        return FaceSessionResponse(**session.to_public_dict())
    except FaceSessionError as exc:
        _raise_session_error(exc)
        raise AssertionError("unreachable")
