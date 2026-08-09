from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from threading import Lock
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.config import settings
from app.core.face_verification_provider import FaceVerificationProvider
from app.modules.face_verification.errors import FaceVerificationError
from app.modules.face_verification.verification_service import (
    FaceVerificationOutput,
    FaceVerificationResult,
)


@dataclass(frozen=True)
class FaceVerificationPipelineOutput:
    verification: FaceVerificationResult
    request_id: str
    reference_source: str = "uploaded_card"
    card_image_path: str | None = None
    webcam_image_path: str | None = None
    portrait_image_path: str | None = None
    webcam_face_path: str | None = None
    result_json_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        data = self.verification.to_dict()
        data.update(
            {
                "success": True,
                "request_id": self.request_id,
                "message": self._build_message(),
                "reference_source": self.reference_source,
            }
        )
        return data

    def _build_message(self) -> str:
        if self.verification.status == "match":
            return "Khuôn mặt trùng khớp với ảnh chân dung trên CCCD."
        if self.verification.status == "review":
            if self.verification.quality_adjusted:
                return (
                    "Độ tương đồng đạt ngưỡng nhưng chất lượng ảnh chưa tối ưu; "
                    "cần chụp lại hoặc kiểm tra thủ công."
                )
            return (
                "Độ tương đồng nằm trong vùng cần kiểm tra; "
                "hãy chụp lại hoặc xác minh thủ công."
            )
        return "Khuôn mặt không trùng khớp với ảnh chân dung trên CCCD."


class FaceVerificationPipeline:
    """Giải mã ảnh, gọi Face Verification và lưu debug khi được bật."""

    _verification_lock = Lock()

    def __init__(
        self,
        provider: FaceVerificationProvider | None = None,
        debug_root: str | Path | None = None,
        max_file_size_mb: int | None = None,
        max_image_pixels: int | None = None,
        save_debug: bool | None = None,
    ) -> None:
        file_size_mb = max_file_size_mb or settings.max_upload_size_mb
        if file_size_mb <= 0:
            raise ValueError("max_file_size_mb phải lớn hơn 0.")

        self._provider = provider
        self.debug_root = Path(debug_root or settings.face_debug_dir)
        self.max_file_size_bytes = file_size_mb * 1024 * 1024
        self.max_image_pixels = max_image_pixels or settings.face_max_image_pixels
        self.save_debug = (
            settings.face_save_debug if save_debug is None else save_debug
        )

    @property
    def provider(self) -> FaceVerificationProvider:
        if self._provider is None:
            self._provider = FaceVerificationProvider.instance()
        return self._provider

    def process(
        self,
        card_image_bytes: bytes,
        webcam_image_bytes: bytes,
    ) -> FaceVerificationPipelineOutput:
        self._validate_file_bytes(card_image_bytes, "card_image")
        self._validate_file_bytes(webcam_image_bytes, "webcam_image")

        card_image = self._decode_image(card_image_bytes, "ảnh CCCD")
        webcam_image = self._decode_image(webcam_image_bytes, "ảnh webcam")
        request_id = self._generate_request_id()

        # ONNX Runtime/InsightFace dùng chung một model. Khóa này tránh hai
        # request chạy đồng thời trên cùng session CPU.
        with self._verification_lock:
            output = self.provider.service.verify(
                card_image=card_image,
                webcam_image=webcam_image,
            )

        debug_paths = self._save_debug_artifacts(
            request_id=request_id,
            card_image=card_image,
            webcam_image=webcam_image,
            verification_output=output,
        )
        return FaceVerificationPipelineOutput(
            verification=output.result,
            request_id=request_id,
            reference_source="uploaded_card",
            card_image_path=debug_paths.get("card_image"),
            webcam_image_path=debug_paths.get("webcam_image"),
            portrait_image_path=debug_paths.get("portrait_image"),
            webcam_face_path=debug_paths.get("webcam_face"),
            result_json_path=debug_paths.get("result_json"),
        )

    def process_from_ocr_paths(
        self,
        *,
        card_image_path: str | Path,
        webcam_image_bytes: bytes,
        portrait_image_path: str | Path | None = None,
    ) -> FaceVerificationPipelineOutput:
        """Đối chiếu selfie với ảnh tham chiếu đã được OCR tạo trước đó."""

        self._validate_file_bytes(webcam_image_bytes, "selfie_image")
        try:
            card_image = self._decode_image_path(
                card_image_path,
                "ảnh thẻ từ OCR",
            )
        except (OSError, ValueError) as exc:
            raise FaceVerificationError(
                "OCR_REFERENCE_IMAGE_INVALID",
                "Ảnh CCCD tham chiếu của phiên OCR không thể sử dụng.",
                status_code=410,
                details={
                    "suggestion": "Vui lòng quét lại mặt trước CCCD."
                },
            ) from exc
        webcam_image = self._decode_image(webcam_image_bytes, "ảnh selfie")

        portrait_image: np.ndarray | None = None
        if portrait_image_path is not None:
            try:
                portrait_image = self._decode_image_path(
                    portrait_image_path,
                    "crop chân dung từ OCR",
                )
            except (OSError, ValueError):
                # Card image là nguồn chuẩn bắt buộc và vẫn dùng được.
                portrait_image = None

        request_id = self._generate_request_id()
        reference_source = "ocr_card_image"

        with self._verification_lock:
            service = self.provider.service
            prepared_verifier = getattr(
                service,
                "verify_prepared_portrait",
                None,
            )
            if portrait_image is not None and callable(prepared_verifier):
                try:
                    output = prepared_verifier(
                        portrait_image,
                        webcam_image,
                        extraction_method="ocr_portrait_crop",
                    )
                    reference_source = "ocr_portrait_crop"
                except FaceVerificationError as exc:
                    # Chỉ fallback khi vấn đề nằm ở chân dung CCCD. Lỗi selfie
                    # hoặc model phải được trả ngay, tránh chạy inference lặp.
                    if not exc.error_code.startswith("CCCD_"):
                        raise
                    output = service.verify(
                        card_image=card_image,
                        webcam_image=webcam_image,
                    )
                    reference_source = "ocr_card_image_fallback"
            else:
                output = service.verify(
                    card_image=card_image,
                    webcam_image=webcam_image,
                )

        debug_paths = self._save_debug_artifacts(
            request_id=request_id,
            card_image=card_image,
            webcam_image=webcam_image,
            verification_output=output,
        )
        return FaceVerificationPipelineOutput(
            verification=output.result,
            request_id=request_id,
            reference_source=reference_source,
            card_image_path=debug_paths.get("card_image"),
            webcam_image_path=debug_paths.get("webcam_image"),
            portrait_image_path=debug_paths.get("portrait_image"),
            webcam_face_path=debug_paths.get("webcam_face"),
            result_json_path=debug_paths.get("result_json"),
        )

    def _validate_file_bytes(self, file_bytes: bytes, file_name: str) -> None:
        if file_bytes is None:
            raise ValueError(f"{file_name} không được là None.")
        if not isinstance(file_bytes, bytes):
            raise TypeError(f"{file_name} phải có kiểu bytes.")
        if not file_bytes:
            raise ValueError(f"{file_name} không được rỗng.")
        if len(file_bytes) > self.max_file_size_bytes:
            max_size_mb = self.max_file_size_bytes / 1024 / 1024
            raise ValueError(
                f"{file_name} vượt quá dung lượng tối đa {max_size_mb:.0f} MB."
            )

    def _decode_image(self, file_bytes: bytes, image_name: str) -> np.ndarray:
        try:
            with Image.open(BytesIO(file_bytes)) as source_image:
                width, height = source_image.size
                if width <= 0 or height <= 0:
                    raise ValueError(f"{image_name} có kích thước không hợp lệ.")
                if width * height > self.max_image_pixels:
                    raise ValueError(
                        f"{image_name} vượt quá giới hạn "
                        f"{self.max_image_pixels:,} pixel."
                    )

                oriented_image = ImageOps.exif_transpose(source_image)
                rgb_image = oriented_image.convert("RGB")
                rgb_array = np.asarray(rgb_image, dtype=np.uint8)
        except (
            UnidentifiedImageError,
            OSError,
            Image.DecompressionBombError,
        ) as exc:
            raise ValueError(
                f"Không thể giải mã {image_name}; hãy dùng JPG, JPEG hoặc PNG hợp lệ."
            ) from exc

        return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    def _decode_image_path(
        self,
        image_path: str | Path,
        image_name: str,
    ) -> np.ndarray:
        path = Path(image_path)
        if not path.is_file():
            raise ValueError(f"Không tìm thấy {image_name}.")
        file_size = path.stat().st_size
        if file_size <= 0:
            raise ValueError(f"{image_name} bị rỗng.")
        if file_size > self.max_file_size_bytes:
            max_size_mb = self.max_file_size_bytes / 1024 / 1024
            raise ValueError(
                f"{image_name} vượt quá dung lượng tối đa {max_size_mb:.0f} MB."
            )
        try:
            file_bytes = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"Không thể đọc {image_name}.") from exc
        return self._decode_image(file_bytes, image_name)

    @staticmethod
    def _generate_request_id() -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{uuid4().hex[:8]}"

    def _save_debug_artifacts(
        self,
        request_id: str,
        card_image: np.ndarray,
        webcam_image: np.ndarray,
        verification_output: FaceVerificationOutput,
    ) -> dict[str, str | None]:
        empty_paths = {
            "card_image": None,
            "webcam_image": None,
            "portrait_image": None,
            "webcam_face": None,
            "result_json": None,
        }
        if not self.save_debug:
            return empty_paths

        request_dir = self.debug_root / request_id
        request_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "card_image": request_dir / "card_image.jpg",
            "webcam_image": request_dir / "webcam_image.jpg",
            "portrait_image": request_dir / "cccd_face.jpg",
            "webcam_face": request_dir / "webcam_face.jpg",
            "result_json": request_dir / "verification_result.json",
        }

        self._write_image(paths["card_image"], card_image)
        self._write_image(paths["webcam_image"], webcam_image)
        self._write_image(
            paths["portrait_image"],
            verification_output.artifacts.cccd_portrait,
        )
        self._write_image(
            paths["webcam_face"],
            verification_output.artifacts.webcam_face,
        )

        result_data = verification_output.result.to_dict()
        result_data.update({"request_id": request_id, "success": True})
        with paths["result_json"].open("w", encoding="utf-8") as file:
            json.dump(result_data, file, ensure_ascii=False, indent=2)

        return {
            key: str(path.resolve())
            for key, path in paths.items()
        }

    @staticmethod
    def _write_image(path: Path, image: np.ndarray) -> None:
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Không thể lưu ảnh debug: {path}")


face_verification_pipeline = FaceVerificationPipeline()
