import cv2
import numpy as np

def check_image_quality(img: np.ndarray, blur_threshold: float = 100.0, dark_threshold: float = 60.0):
    """
    Hàm kiểm tra chất lượng ảnh đầu vào (Task 9 - Reject Image)
    Trả về dict chứa kết quả hợp lệ hay không và lý do từ chối.
    """
    # 1. Chuyển ảnh sang dạng xám (Grayscale) để dễ tính toán
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. TÍNH ĐỘ MỜ (Blur) bằng phương sai Laplacian
    # Lấy phương sai của đạo hàm bậc 2. Nếu < 100 thường là ảnh bị rung tay, nhòe.
    blur_score = cv2.Laplacian(gray_img, cv2.CV_64F).var()
    is_blurry = blur_score < blur_threshold
    
    # 3. TÍNH ĐỘ SÁNG (Brightness) bằng trung bình pixel
    # Thang điểm từ 0 (Đen thui) đến 255 (Trắng toát). Nếu < 60 là quá tối.
    brightness_score = np.mean(gray_img)
    is_too_dark = brightness_score < dark_threshold
    
    # Xác định lý do nếu bị từ chối
    reason = "Hợp lệ"
    if is_blurry and is_too_dark:
        reason = "Ảnh quá mờ và quá tối"
    elif is_blurry:
        reason = "Ảnh quá mờ (Blurry)"
    elif is_too_dark:
        reason = "Ảnh thiếu sáng (Too dark)"

    return {
        "is_valid": not (is_blurry or is_too_dark),
        "blur_score": blur_score,
        "brightness_score": brightness_score,
        "reason": reason
    }