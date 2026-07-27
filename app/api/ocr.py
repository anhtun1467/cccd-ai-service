import cv2
from fastapi import APIRouter, File, UploadFile, HTTPException

from app.schemas.response import ApiResponse
from app.services.ocr_pipeline import ocr_pipeline_service
from app.utils.file_utils import save_upload_file

# Import hàm Reject Image mà bạn vừa tạo ở Bước 2
from app.utils.image_validator import check_image_quality

router = APIRouter(prefix="/ocr", tags=["OCR"])


@router.post("/cccd", response_model=ApiResponse)
async def ocr_cccd(file: UploadFile = File(...)) -> ApiResponse:
    # 1. Lưu file ảnh vào hệ thống (giữ nguyên logic cũ)
    image_path = await save_upload_file(file)

    # ==========================================
    # TASK 9: CƠ CHẾ REJECT IMAGE
    # ==========================================
    # 2. Đọc ảnh vừa lưu lên bằng OpenCV
    img = cv2.imread(str(image_path))
    if img is None:
        raise HTTPException(status_code=400, detail="Không thể đọc được file ảnh đầu vào.")

    # 3. Chấm điểm chất lượng ảnh
    quality = check_image_quality(img)
    print(f"\n[TASK 9 - KIỂM DUYỆT ẢNH] Blur: {quality['blur_score']:.2f} | Sáng: {quality['brightness_score']:.2f}")

    # 4. Quyết định từ chối (Reject) nếu ảnh kém chất lượng
    if not quality["is_valid"]:
        raise HTTPException(
            status_code=400,
            detail=f"Hình ảnh bị từ chối: {quality['reason']}. Vui lòng chụp lại rõ nét và đủ ánh sáng hơn!"
        )
    # ==========================================

    # 5. Vượt qua kiểm duyệt -> Đưa vào luồng xử lý OCR AI
    result = ocr_pipeline_service.process_cccd_image(image_path)

    return ApiResponse(
        success=True,
        message="Nhận ảnh CCCD thành công",
        data=result,
    )