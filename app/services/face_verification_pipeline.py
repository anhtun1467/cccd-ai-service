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
    Kết quả trả về từ pipeline Face Verification.
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
            return "Khuôn mặt trùng khớp với ảnh chân dung trên CCCD."

        if self.verification.status == "review":
            return (
                "Độ tương đồng chưa đủ rõ ràng. "
                "Cần kiểm tra thủ công hoặc chụp lại ảnh."
            )

        return "Khuôn mặt không trùng khớp với ảnh chân dung trên CCCD."


class FaceVerificationPipeline:
    """
    Pipeline xử lý ảnh CCCD và ảnh webcam.

    Nhiệm vụ:
        1. Kiểm tra dữ liệu ảnh.
        2. Giải mã bytes thành ảnh OpenCV.
        3. Gọi FaceVerificationService.
        4. Lưu artifacts phục vụ debug.
        5. Trả kết quả chuẩn hóa cho API.
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
                "max_file_size_mb phải lớn hơn 0."
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
        Xử lý xác minh khuôn mặt từ dữ liệu ảnh dạng bytes.

        Args:
            card_image_bytes:
                Nội dung file ảnh CCCD.

            webcam_image_bytes:
                Nội dung file ảnh chụp từ webcam.

        Returns:
            FaceVerificationPipelineOutput.

        Raises:
            ValueError:
                Khi dữ liệu file không hợp lệ hoặc OpenCV
                không giải mã được ảnh.

            RuntimeError:
                Khi quá trình phát hiện hoặc xác minh khuôn mặt
                không thực hiện được.
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
            image_name="ảnh CCCD",
        )

        webcam_image = self._decode_image(
            webcam_image_bytes,
            image_name="ảnh webcam",
        )

        request_id = self._generate_request_id()

        # InsightFace/ONNX Runtime có thể được gọi đồng thời từ nhiều
        # request. Lock giúp giai đoạn đầu vận hành ổn định trên CPU.
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
                f"{file_name} không được là None."
            )

        if not isinstance(file_bytes, bytes):
            raise TypeError(
                f"{file_name} phải có kiểu bytes."
            )

        if len(file_bytes) == 0:
            raise ValueError(
                f"{file_name} không được rỗng."
            )

        if len(file_bytes) > self.max_file_size_bytes:
            max_size_mb = (
                self.max_file_size_bytes
                / 1024
                / 1024
            )

            raise ValueError(
                f"{file_name} vượt quá dung lượng tối đa "
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
                f"Không thể giải mã {image_name}. "
                "Hãy sử dụng file JPG, JPEG hoặc PNG hợp lệ."
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
                f"Không thể lưu ảnh debug: {path}"
            )


face_verification_pipeline = FaceVerificationPipeline()
