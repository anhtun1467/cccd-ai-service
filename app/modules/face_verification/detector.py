from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from insightface.app import FaceAnalysis


@dataclass(frozen=True)
class DetectedFace:
    """
    Kết quả phát hiện một khuôn mặt.

    Attributes:
        bbox:
            Bounding box theo định dạng:
            [x_min, y_min, x_max, y_max].

        score:
            Độ tin cậy của detector.

        landmarks:
            Năm điểm mốc khuôn mặt:
            mắt trái, mắt phải, mũi,
            khóe miệng trái, khóe miệng phải.
    """

    bbox: np.ndarray
    score: float
    landmarks: np.ndarray | None

    @property
    def x1(self) -> int:
        return int(self.bbox[0])

    @property
    def y1(self) -> int:
        return int(self.bbox[1])

    @property
    def x2(self) -> int:
        return int(self.bbox[2])

    @property
    def y2(self) -> int:
        return int(self.bbox[3])

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height


class InsightFaceDetector:
    """
    Phát hiện khuôn mặt bằng detector của InsightFace.

    Giai đoạn đầu sử dụng CPUExecutionProvider
    để bảo đảm môi trường ổn định.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        detection_size: tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.60,
        providers: Sequence[str] | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold phải nằm trong khoảng từ 0 đến 1."
            )

        if detection_size[0] <= 0 or detection_size[1] <= 0:
            raise ValueError("detection_size phải lớn hơn 0.")

        self.model_name = model_name
        self.detection_size = detection_size
        self.confidence_threshold = confidence_threshold

        self.providers = list(
            providers or ["CPUExecutionProvider"]
        )

        self._face_analysis = FaceAnalysis(
            name=self.model_name,
            allowed_modules=["detection"],
            providers=self.providers,
        )

        # ctx_id = -1: chạy CPU
        self._face_analysis.prepare(
            ctx_id=-1,
            det_size=self.detection_size,
        )

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        """
        Phát hiện tất cả khuôn mặt đạt ngưỡng confidence.

        Args:
            image:
                Ảnh OpenCV định dạng BGR.

        Returns:
            Danh sách DetectedFace, được sắp xếp
            theo diện tích giảm dần.
        """

        self._validate_image(image)

        raw_faces = self._face_analysis.get(image)

        detected_faces: list[DetectedFace] = []

        for face in raw_faces:
            score = float(face.det_score)

            if score < self.confidence_threshold:
                continue

            bbox = np.asarray(face.bbox, dtype=np.float32)

            landmarks = None

            if getattr(face, "kps", None) is not None:
                landmarks = np.asarray(
                    face.kps,
                    dtype=np.float32,
                )

            detected_faces.append(
                DetectedFace(
                    bbox=bbox,
                    score=score,
                    landmarks=landmarks,
                )
            )

        detected_faces.sort(
            key=lambda detected_face: detected_face.area,
            reverse=True,
        )

        return detected_faces

    def detect_single(
        self,
        image: np.ndarray,
    ) -> DetectedFace | None:
        """
        Trả về khuôn mặt lớn nhất trong ảnh.

        Nếu không tìm thấy khuôn mặt thì trả về None.
        """

        faces = self.detect(image)

        if not faces:
            return None

        return faces[0]

    @staticmethod
    def crop_face(
        image: np.ndarray,
        face: DetectedFace,
        margin_ratio: float = 0.15,
    ) -> np.ndarray:
        """
        Cắt khuôn mặt khỏi ảnh, có thêm phần lề.

        Args:
            image:
                Ảnh gốc BGR.

            face:
                Kết quả phát hiện khuôn mặt.

            margin_ratio:
                Tỉ lệ lề thêm xung quanh khuôn mặt.
        """

        if margin_ratio < 0:
            raise ValueError("margin_ratio không được nhỏ hơn 0.")

        image_height, image_width = image.shape[:2]

        margin_x = int(face.width * margin_ratio)
        margin_y = int(face.height * margin_ratio)

        x1 = max(0, face.x1 - margin_x)
        y1 = max(0, face.y1 - margin_y)
        x2 = min(image_width, face.x2 + margin_x)
        y2 = min(image_height, face.y2 + margin_y)

        if x2 <= x1 or y2 <= y1:
            raise ValueError("Bounding box khuôn mặt không hợp lệ.")

        return image[y1:y2, x1:x2].copy()

    @staticmethod
    def draw_faces(
        image: np.ndarray,
        faces: Sequence[DetectedFace],
    ) -> np.ndarray:
        """
        Vẽ bounding box, confidence và landmarks.
        """

        output = image.copy()

        for index, face in enumerate(faces, start=1):
            cv2.rectangle(
                output,
                (face.x1, face.y1),
                (face.x2, face.y2),
                (0, 255, 0),
                2,
            )

            label = f"Face {index}: {face.score:.3f}"

            text_y = max(25, face.y1 - 10)

            cv2.putText(
                output,
                label,
                (face.x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if face.landmarks is not None:
                for point in face.landmarks:
                    point_x = int(point[0])
                    point_y = int(point[1])

                    cv2.circle(
                        output,
                        (point_x, point_y),
                        3,
                        (0, 0, 255),
                        -1,
                    )

        return output

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if image is None:
            raise ValueError("Ảnh đầu vào không được là None.")

        if not isinstance(image, np.ndarray):
            raise TypeError("Ảnh đầu vào phải là numpy.ndarray.")

        if image.size == 0:
            raise ValueError("Ảnh đầu vào rỗng.")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "Ảnh đầu vào phải có định dạng BGR với 3 kênh màu."
            )