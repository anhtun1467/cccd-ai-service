from __future__ import annotations

import logging
from pathlib import Path

import cv2
from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.schemas.response import ApiResponse
from app.services.ocr_pipeline import ocr_pipeline_service
from app.utils.file_utils import save_upload_file
from app.utils.image_validator import check_image_quality

router = APIRouter(prefix="/ocr", tags=["OCR"])
logger = logging.getLogger(__name__)


@router.post("/cccd", response_model=ApiResponse)
async def ocr_cccd(file: UploadFile = File(...)) -> ApiResponse:
    # Lưu ảnh và chuyển đường dẫn str thành Path
    saved_path = await save_upload_file(file)
    image_path = Path(saved_path)

    image = cv2.imread(str(image_path))

    if image is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Không thể đọc được file ảnh đầu vào. "
                "Vui lòng chọn ảnh JPG hoặc PNG hợp lệ."
            ),
        )

    try:
        quality = check_image_quality(image)
    except Exception as exc:
        logger.exception("Lỗi kiểm tra chất lượng ảnh: %s", image_path)
        raise HTTPException(
            status_code=500,
            detail="Không thể kiểm tra chất lượng ảnh.",
        ) from exc

    blur_score = float(quality.get("blur_score", 0.0))
    brightness_score = float(quality.get("brightness_score", 0.0))
    is_valid = bool(quality.get("is_valid", False))
    reason = quality.get("reason") or "Ảnh không đạt yêu cầu chất lượng"

    logger.info(
        "Kiểm duyệt ảnh %s | Blur: %.2f | Brightness: %.2f | Valid: %s",
        image_path.name,
        blur_score,
        brightness_score,
        is_valid,
    )

    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Hình ảnh CCCD không đạt yêu cầu.",
                "reason": reason,
                "blur_score": round(blur_score, 2),
                "brightness_score": round(brightness_score, 2),
                "suggestion": (
                    "Vui lòng chụp lại ảnh rõ nét, đủ sáng "
                    "và giữ CCCD ổn định."
                ),
            },
        )

    try:
        result = await run_in_threadpool(
            ocr_pipeline_service.process_cccd_image,
            image_path,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Xử lý OCR thất bại: %s", image_path)
        raise HTTPException(
            status_code=500,
            detail="Không thể xử lý ảnh CCCD.",
        ) from exc

    return ApiResponse(
        success=True,
        message="Nhận dạng CCCD thành công",
        data=result,
    )