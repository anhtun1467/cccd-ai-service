from fastapi import APIRouter, File, UploadFile

from app.schemas.response import ApiResponse
from app.services.ocr_pipeline import ocr_pipeline_service
from app.utils.file_utils import save_upload_file


router = APIRouter(prefix="/ocr", tags=["OCR"])


@router.post("/cccd", response_model=ApiResponse)
async def ocr_cccd(file: UploadFile = File(...)) -> ApiResponse:
    image_path = await save_upload_file(file)
    result = ocr_pipeline_service.process_cccd_image(image_path)

    return ApiResponse(
        success=True,
        message="Nhận ảnh CCCD thành công",
        data=result,
    )