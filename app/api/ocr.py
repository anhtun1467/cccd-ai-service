import cv2
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.response import ApiResponse
from app.services.ocr_pipeline import ocr_pipeline_service
from app.utils.file_utils import save_upload_file

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

    # Không từ chối ảnh chỉ dựa trên Laplacian của toàn khung hình.
    # Nền trơn, ảnh đã resize hoặc camera làm mịn có thể cho điểm thấp dù
    # chữ trên thẻ vẫn đọc được. Pipeline sẽ đánh giá lại vùng thẻ sau khi
    # làm phẳng và kết hợp điểm ảnh với bằng chứng OCR thực tế.
    quality = check_image_quality(
        img,
        blur_threshold=0.0,
        dark_threshold=60.0,
    )
    print(f"\n[TASK 9 - KIỂM DUYỆT ẢNH] Blur: {quality['blur_score']:.2f} | Sáng: {quality['brightness_score']:.2f}")

    # 4. Ở đầu vào chỉ chặn ảnh thiếu sáng rõ rệt. Độ mờ được hoãn quyết
    # định đến sau OCR để tránh loại nhầm ảnh hơi mờ nhưng vẫn đọc được.
    if not quality["is_valid"]:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Hình ảnh CCCD không đạt yêu cầu.",
                "error_code": quality["error_code"],
                "reason": quality["reason"],
                "blur_score": quality["blur_score"],
                "brightness_score": quality["brightness_score"],
                "suggestion": quality["suggestion"],
            },
        )
    # ==========================================

    # 5. Vượt qua kiểm duyệt -> Đưa vào luồng xử lý OCR AI
    result = ocr_pipeline_service.process_cccd_image(image_path)

    if result.get("status") == "OCR_FAILED":
        metadata = result.get("metadata", {})
        rejection = metadata.get("rejection", {})
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Hình ảnh CCCD không đạt yêu cầu.",
                "error_code": rejection.get(
                    "errorCode",
                    "OCR_FAILED",
                ),
                "reason": rejection.get(
                    "reason",
                    result.get("message", "OCR CCCD thất bại"),
                ),
                "blur_score": rejection.get("blurScore"),
                "brightness_score": rejection.get("brightnessScore"),
                "card_count": rejection.get("cardCount"),
                "suggestion": rejection.get(
                    "suggestion",
                    "Vui lòng chụp lại một CCCD rõ nét và đủ sáng.",
                ),
            },
        )

    return ApiResponse(
        success=True,
        message="Nhận ảnh CCCD thành công",
        data=result,
    )
