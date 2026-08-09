import os
import uuid
from pathlib import Path

from fastapi import UploadFile

from app.core.config import settings
from app.core.exceptions import BadRequestException


ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def ensure_storage_dirs() -> None:
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.face_session_dir).mkdir(parents=True, exist_ok=True)


def get_file_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def is_allowed_image(filename: str) -> bool:
    return get_file_extension(filename) in ALLOWED_IMAGE_EXTENSIONS


def generate_unique_filename(original_filename: str) -> str:
    extension = get_file_extension(original_filename)
    return f"{uuid.uuid4().hex}{extension}"


async def save_upload_file(file: UploadFile) -> str:
    if not file.filename:
        raise BadRequestException("Tên file không hợp lệ")

    if not is_allowed_image(file.filename):
        raise BadRequestException("Chỉ hỗ trợ ảnh JPG, JPEG hoặc PNG")

    content = await file.read()

    max_size_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_size_bytes:
        raise BadRequestException(f"File vượt quá {settings.max_upload_size_mb}MB")

    unique_filename = generate_unique_filename(file.filename)
    save_path = os.path.join(settings.upload_dir, unique_filename)

    with open(save_path, "wb") as buffer:
        buffer.write(content)

    return save_path
