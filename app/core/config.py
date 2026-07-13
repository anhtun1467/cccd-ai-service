from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "CCCD AI Service"
    app_version: str = "1.0.0"
    debug: bool = True

    upload_dir: str = "storage/uploads"
    output_dir: str = "storage/outputs"
    max_upload_size_mb: int = 10

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