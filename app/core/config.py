from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "CCCD AI Service"
    app_version: str = "1.0.0"
    debug: bool = True

    upload_dir: str = "storage/uploads"
    output_dir: str = "storage/outputs"
    max_upload_size_mb: int = 10

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
