from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from uuid import uuid4

import cv2
import numpy as np

from app.core.face_verification_provider import (
    FaceVerificationProvider,
)
from app.modules.face_verification.verification_service import (
    FaceVerificationOutput,
    FaceVerificationResult,
)


@dataclass(frozen=True)
class FaceVerificationPipelineOutput:
    """
    K?t qu? tr? v? t? pipeline Face Verification.
    """

    verification: FaceVerificationResult
    request_id: str
    card_image_path: str | None
    webcam_image_path: str | None
    portrait_image_path: str | None
    result_json_path: str | None

    def to_dict(self) -> dict[str, object]:
        data = self.verification.to_dict()

        data.update(
            {
                "success": True,
                "request_id": self.request_id,
                "message": self._build_message(),
            }
        )

        return data

    def _build_message(self) -> str:
        if self.verification.status == "match":
            return "Khuôn m?t trùng kh?p v?i ?nh chân dung trên CCCD."

        if self.verification.status == "review":
            return (
                "Ð? tuong d?ng chua d? rõ ràng. "
                "C?n ki?m tra th? công ho?c ch?p l?i ?nh."
            )

        return "Khuôn m?t không trùng kh?p v?i ?nh chân dung trên CCCD."


class FaceVerificationPipeline:
    """
    Pipeline x? lý ?nh CCCD và ?nh webcam.

    Nhi?m v?:
        1. Ki?m tra d? li?u ?nh.
        2. Gi?i mã bytes thành ?nh OpenCV.
        3. G?i FaceVerificationService.
        4. Luu artifacts ph?c v? debug.
        5. Tr? k?t qu? chu?n hóa cho API.
    """

    _verification_lock = Lock()

    def __init__(
        self,
        provider: FaceVerificationProvider | None = None,
        debug_root: str | Path = "storage/debug/face_verification_api",
        max_file_size_mb: int = 10,
        save_debug: bool = True,
    ) -> None:
        if max_file_size_mb <= 0:
            raise ValueError(
                "max_file_size_mb ph?i l?n hon 0."
            )

        self.provider = (
            provider
            or FaceVerificationProvider.instance()
        )

        self.debug_root = Path(debug_root)
        self.max_file_size_bytes = (
            max_file_size_mb * 1024 * 1024
        )
        self.save_debug = save_debug

    def process(
        self,
        card_image_bytes: bytes,
        webcam_image_bytes: bytes,
    ) -> FaceVerificationPipelineOutput:
        """
        X? lý xác minh khuôn m?t t? d? li?u ?nh d?ng bytes.

        Args:
            card_image_bytes:
                N?i dung file ?nh CCCD.

            webcam_image_bytes:
                N?i dung file ?nh ch?p t? webcam.

        Returns:
            FaceVerificationPipelineOutput.

        Raises:
            ValueError:
                Khi d? li?u file không h?p l? ho?c OpenCV
                không gi?i mã du?c ?nh.

            RuntimeError:
                Khi quá trình phát hi?n ho?c xác minh khuôn m?t
                không th?c hi?n du?c.
        """

        self._validate_file_bytes(
            card_image_bytes,
            file_name="card_image",
        )

        self._validate_file_bytes(
            webcam_image_bytes,
            file_name="webcam_image",
        )

        card_image = self._decode_image(
            card_image_bytes,
            image_name="?nh CCCD",
        )

        webcam_image = self._decode_image(
            webcam_image_bytes,
            image_name="?nh webcam",
        )

        request_id = self._generate_request_id()

        # InsightFace/ONNX Runtime có th? du?c g?i d?ng th?i t? nhi?u
        # request. Lock giúp giai do?n d?u v?n hành ?n d?nh trên CPU.
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
            card_image_path=debug_paths.get("card_image"),
            webcam_image_path=debug_paths.get("webcam_image"),
            portrait_image_path=debug_paths.get("portrait_image"),
            result_json_path=debug_paths.get("result_json"),
        )

    def _validate_file_bytes(
        self,
        file_bytes: bytes,
        file_name: str,
    ) -> None:
        if file_bytes is None:
            raise ValueError(
                f"{file_name} không du?c là None."
            )

        if not isinstance(file_bytes, bytes):
            raise TypeError(
                f"{file_name} ph?i có ki?u bytes."
            )

        if len(file_bytes) == 0:
            raise ValueError(
                f"{file_name} không du?c r?ng."
            )

        if len(file_bytes) > self.max_file_size_bytes:
            max_size_mb = (
                self.max_file_size_bytes
                / 1024
                / 1024
            )

            raise ValueError(
                f"{file_name} vu?t quá dung lu?ng t?i da "
                f"{max_size_mb:.0f} MB."
            )

    @staticmethod
    def _decode_image(
        file_bytes: bytes,
        image_name: str,
    ) -> np.ndarray:
        image_array = np.frombuffer(
            file_bytes,
            dtype=np.uint8,
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR,
        )

        if image is None or image.size == 0:
            raise ValueError(
                f"Không th? gi?i mã {image_name}. "
                "Hãy s? d?ng file JPG, JPEG ho?c PNG h?p l?."
            )

        return image

    @staticmethod
    def _generate_request_id() -> str:
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        random_suffix = uuid4().hex[:8]

        return f"{timestamp}_{random_suffix}"

    def _save_debug_artifacts(
        self,
        request_id: str,
        card_image: np.ndarray,
        webcam_image: np.ndarray,
        verification_output: FaceVerificationOutput,
    ) -> dict[str, str | None]:
        if not self.save_debug:
            return {
                "card_image": None,
                "webcam_image": None,
                "portrait_image": None,
                "result_json": None,
            }

        request_dir = (
            self.debug_root
            / request_id
        )

        request_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        card_path = request_dir / "card_image.jpg"
        webcam_path = request_dir / "webcam_image.jpg"
        portrait_path = request_dir / "cccd_portrait.jpg"
        result_path = request_dir / "verification_result.json"

        self._write_image(
            card_path,
            card_image,
        )

        self._write_image(
            webcam_path,
            webcam_image,
        )

        self._write_image(
            portrait_path,
            verification_output.artifacts.cccd_portrait,
        )

        result_data = verification_output.result.to_dict()

        result_data.update(
            {
                "request_id": request_id,
                "success": True,
            }
        )

        with result_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                result_data,
                file,
                ensure_ascii=False,
                indent=2,
            )

        return {
            "card_image": str(card_path.resolve()),
            "webcam_image": str(webcam_path.resolve()),
            "portrait_image": str(portrait_path.resolve()),
            "result_json": str(result_path.resolve()),
        }

    @staticmethod
    def _write_image(
        path: Path,
        image: np.ndarray,
    ) -> None:
        success = cv2.imwrite(
            str(path),
            image,
        )

        if not success:
            raise RuntimeError(
                f"Không th? luu ?nh debug: {path}"
            )


face_verification_pipeline = FaceVerificationPipeline()

