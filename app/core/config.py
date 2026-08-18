from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "CCCD AI Service"
    app_version: str = "1.0.0"
    debug: bool = True

    upload_dir: str = "storage/uploads"
    output_dir: str = "storage/outputs"
    max_upload_size_mb: int = 10

    # QR Fast Path - thử đọc QR ngay sau khi vùng thẻ được làm phẳng
    # để xác định chiều trước full-card OCR và bỏ qua các field OCR
    # đã có dữ liệu xác nhận từ QR.
    # Ngân sách thời gian là mềm vì OpenCV không hỗ trợ hủy một lần
    # detect/decode đang chạy giữa chừng.
    qr_fast_path_enabled: bool = True
    qr_decode_budget_ms: float = 120.0
    qr_skip_confirmed_field_ocr: bool = True

    # Face Verification - InsightFace buffalo_l / ArcFace 512-D.
    face_model_name: str = "buffalo_l"
    face_execution_provider: str = "CPUExecutionProvider"
    face_detection_size: int = 640
    face_detection_confidence: float = 0.50
    face_match_threshold: float = 0.50
    face_review_threshold: float = 0.40
    face_max_image_pixels: int = 20_000_000

    # Ảnh CCCD và selfie là dữ liệu nhạy cảm; không lưu mặc định.
    face_save_debug: bool = False
    face_debug_dir: str = "storage/debug/face_verification_api"

    # Phiên nối kết quả OCR với Face Verification.
    # Client chỉ giữ một session id ngẫu nhiên, không được nhận
    # hoặc tự gửi đường dẫn ảnh CCCD.
    face_session_dir: str = "storage/face_sessions"
    face_session_ttl_seconds: int = 1800
    face_session_max_attempts: int = 5
    face_session_expired_retention_seconds: int = 86400
    face_session_lease_timeout_seconds: int = 300

    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()