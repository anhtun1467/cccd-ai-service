from __future__ import annotations

import json
import os
import re
import secrets
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Iterator, Literal
from uuid import uuid4

from app.core.config import settings


FaceSessionStatus = Literal[
    "active",
    "verified",
    "exhausted",
    "expired",
    "cancelled",
]

CaptureSource = Literal["camera", "upload"]

SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{32,128}$")


class FaceSessionError(RuntimeError):
    """Lỗi nghiệp vụ của phiên nối OCR với Face Verification."""

    def __init__(
        self,
        error_code: str,
        message: str,
        *,
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)

    def to_data(self) -> dict[str, Any]:
        return {
            "errorCode": self.error_code,
            **self.details,
        }


@dataclass(frozen=True)
class FaceSession:
    """Bản ghi tối thiểu liên kết một kết quả OCR với bước đối chiếu mặt."""

    version: int
    session_id: str
    ocr_request_id: str
    card_image_path: str
    portrait_image_path: str | None
    created_at: str
    updated_at: str
    expires_at: str
    max_attempts: int
    attempts_used: int = 0
    status: FaceSessionStatus = "active"
    last_verification_status: str | None = None
    last_error_code: str | None = None
    last_capture_source: CaptureSource | None = None

    @property
    def remaining_attempts(self) -> int:
        return max(0, self.max_attempts - self.attempts_used)

    def is_expired(self, now: datetime | None = None) -> bool:
        current_time = now or datetime.now(timezone.utc)
        return current_time >= _parse_utc_datetime(self.expires_at)

    def to_public_dict(self, now: datetime | None = None) -> dict[str, Any]:
        effective_status: FaceSessionStatus = self.status
        if self.is_expired(now) and self.status == "active":
            effective_status = "expired"

        return {
            "session_id": self.session_id,
            "ocr_request_id": self.ocr_request_id,
            "status": effective_status,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "max_attempts": self.max_attempts,
            "attempts_used": self.attempts_used,
            "remaining_attempts": self.remaining_attempts,
            "can_verify": bool(
                effective_status == "active" and self.remaining_attempts > 0
            ),
            "verify_endpoint": "/api/face-verification/verify-from-ocr",
            "last_verification_status": self.last_verification_status,
            "last_error_code": self.last_error_code,
            "last_capture_source": self.last_capture_source,
        }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_utc_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_utc_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class FaceSessionStore:
    """Kho phiên dạng JSON, tồn tại qua lần reload Uvicorn.

    Client chỉ nhận ``session_id`` ngẫu nhiên. Đường dẫn ảnh CCCD/portrait
    luôn được giữ ở phía server và phải nằm trong các thư mục đầu ra OCR.
    """

    RECORD_VERSION = 1

    def __init__(
        self,
        *,
        session_dir: str | Path | None = None,
        ttl_seconds: int | None = None,
        max_attempts: int | None = None,
        project_root: str | Path | None = None,
        expired_retention_seconds: int | None = None,
        lease_timeout_seconds: int | None = None,
    ) -> None:
        self.project_root = Path(
            project_root or Path(__file__).resolve().parents[2]
        ).resolve()
        self.session_dir = self._absolute_path(
            session_dir or settings.face_session_dir
        )
        self.output_root = self._absolute_path(settings.output_dir)
        self.debug_root = self._absolute_path(Path("storage") / "debug")
        self.ttl_seconds = int(
            ttl_seconds
            if ttl_seconds is not None
            else settings.face_session_ttl_seconds
        )
        self.max_attempts = int(
            max_attempts
            if max_attempts is not None
            else settings.face_session_max_attempts
        )
        self.expired_retention_seconds = int(
            expired_retention_seconds
            if expired_retention_seconds is not None
            else settings.face_session_expired_retention_seconds
        )
        self.lease_timeout_seconds = int(
            lease_timeout_seconds
            if lease_timeout_seconds is not None
            else settings.face_session_lease_timeout_seconds
        )

        if self.ttl_seconds <= 0:
            raise ValueError("face_session_ttl_seconds phải lớn hơn 0.")
        if self.max_attempts <= 0:
            raise ValueError("face_session_max_attempts phải lớn hơn 0.")
        if self.expired_retention_seconds < 0:
            raise ValueError(
                "face_session_expired_retention_seconds không được âm."
            )
        if self.lease_timeout_seconds <= 0:
            raise ValueError(
                "face_session_lease_timeout_seconds phải lớn hơn 0."
            )

        self._lock = RLock()
        self._verification_in_progress: set[str] = set()

    def create_from_ocr_result(
        self,
        ocr_result: dict[str, Any],
        *,
        ocr_request_id: str,
    ) -> FaceSession:
        """Tạo phiên từ chính ảnh thẻ/portrait mà OCR vừa sinh ra."""

        if ocr_result.get("status") not in {"OCR_SUCCESS", "OCR_PARTIAL"}:
            raise FaceSessionError(
                "OCR_RESULT_NOT_USABLE_FOR_FACE",
                "Kết quả OCR chưa đủ điều kiện tạo phiên đối chiếu khuôn mặt.",
                status_code=422,
            )

        metadata = ocr_result.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        card_image_value = metadata.get("cardImage")
        card_image = self._validate_reference_image(
            card_image_value,
            allowed_root=self.output_root,
            field_name="cardImage",
        )

        portrait_image: Path | None = None
        portrait = ocr_result.get("portrait")
        if isinstance(portrait, dict):
            portrait_value = (
                portrait.get("rawImagePath") or portrait.get("imagePath")
            )
            if portrait_value:
                try:
                    portrait_image = self._validate_reference_image(
                        portrait_value,
                        allowed_root=self.debug_root,
                        field_name="portraitImage",
                    )
                except FaceSessionError:
                    # Ảnh thẻ đã làm phẳng vẫn là nguồn dự phòng đầy đủ.
                    portrait_image = None

        now = _utc_now()
        session = FaceSession(
            version=self.RECORD_VERSION,
            session_id=secrets.token_urlsafe(32),
            ocr_request_id=self._validate_ocr_request_id(ocr_request_id),
            card_image_path=self._to_project_relative(card_image),
            portrait_image_path=(
                self._to_project_relative(portrait_image)
                if portrait_image is not None
                else None
            ),
            created_at=_format_utc_datetime(now),
            updated_at=_format_utc_datetime(now),
            expires_at=_format_utc_datetime(
                now + timedelta(seconds=self.ttl_seconds)
            ),
            max_attempts=self.max_attempts,
        )

        with self._lock:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self._cleanup_old_sessions_unlocked(now)
            self._write_unlocked(session)
        return session

    def get_public(self, session_id: str) -> FaceSession:
        """Đọc trạng thái phiên; phiên vừa hết hạn vẫn được mô tả rõ."""

        with self._lock:
            session = self._read_unlocked(session_id)
            return self._mark_expired_unlocked(session)

    def require_verifiable(self, session_id: str) -> FaceSession:
        """Trả phiên đang hoạt động hoặc ném lỗi HTTP-friendly."""

        with self._lock:
            session = self._read_unlocked(session_id)
            session = self._mark_expired_unlocked(session)

            if session.status == "expired":
                raise FaceSessionError(
                    "FACE_SESSION_EXPIRED",
                    "Phiên OCR đã hết hạn; vui lòng quét lại CCCD.",
                    status_code=410,
                    details={
                        "expiresAt": session.expires_at,
                        "session": session.to_public_dict(),
                    },
                )
            if session.status == "cancelled":
                raise FaceSessionError(
                    "FACE_SESSION_CANCELLED",
                    "Phiên OCR đã bị hủy.",
                    status_code=410,
                    details={"session": session.to_public_dict()},
                )
            if session.status == "verified":
                raise FaceSessionError(
                    "FACE_SESSION_ALREADY_VERIFIED",
                    "Phiên OCR này đã đối chiếu thành công.",
                    status_code=409,
                    details={"session": session.to_public_dict()},
                )
            if session.status == "exhausted" or session.remaining_attempts <= 0:
                raise FaceSessionError(
                    "FACE_SESSION_ATTEMPTS_EXHAUSTED",
                    "Phiên OCR đã hết số lần đối chiếu; vui lòng quét lại CCCD.",
                    status_code=429,
                    details={"session": session.to_public_dict()},
                )

            self.resolve_card_image_path(session)
            return session

    @contextmanager
    def verification_lease(self, session_id: str) -> Iterator[FaceSession]:
        """Chặn hai request đồng thời dùng cùng một phiên trong một worker."""

        session = self.require_verifiable(session_id)
        lease_path: Path | None = None
        lease_token: str | None = None
        with self._lock:
            if session.session_id in self._verification_in_progress:
                raise FaceSessionError(
                    "FACE_SESSION_BUSY",
                    "Phiên OCR đang được đối chiếu bởi một request khác.",
                    status_code=409,
                )
            lease_path, lease_token = self._acquire_lease_file_unlocked(
                session.session_id
            )
            self._verification_in_progress.add(session.session_id)

        try:
            yield session
        finally:
            with self._lock:
                self._verification_in_progress.discard(session.session_id)
                if lease_path is not None and lease_token is not None:
                    self._release_lease_file_unlocked(
                        lease_path,
                        lease_token,
                    )

    def record_attempt(
        self,
        session_id: str,
        *,
        verification_status: str,
        capture_source: CaptureSource,
        error_code: str | None = None,
    ) -> FaceSession:
        """Ghi nhận một lần model đã xử lý selfie."""

        if capture_source not in {"camera", "upload"}:
            raise ValueError("capture_source không hợp lệ.")

        with self._lock:
            session = self._read_unlocked(session_id)
            session = self._mark_expired_unlocked(session)
            if session.status != "active":
                return session

            attempts_used = min(
                session.max_attempts,
                session.attempts_used + 1,
            )
            if verification_status == "match":
                next_status: FaceSessionStatus = "verified"
            elif attempts_used >= session.max_attempts:
                next_status = "exhausted"
            else:
                next_status = "active"

            updated = replace(
                session,
                updated_at=_format_utc_datetime(_utc_now()),
                attempts_used=attempts_used,
                status=next_status,
                last_verification_status=verification_status,
                last_error_code=error_code,
                last_capture_source=capture_source,
            )
            self._write_unlocked(updated)
            return updated

    def cancel(self, session_id: str) -> FaceSession:
        with self._lock:
            session = self._read_unlocked(session_id)
            if session.status == "cancelled":
                return session
            updated = replace(
                session,
                status="cancelled",
                updated_at=_format_utc_datetime(_utc_now()),
            )
            self._write_unlocked(updated)
            return updated

    def resolve_card_image_path(self, session: FaceSession) -> Path:
        return self._resolve_stored_reference(
            session.card_image_path,
            allowed_root=self.output_root,
            error_code="OCR_CARD_IMAGE_MISSING",
            message=(
                "Ảnh CCCD của phiên OCR không còn tồn tại; vui lòng quét lại."
            ),
        )

    def resolve_portrait_image_path(self, session: FaceSession) -> Path | None:
        if not session.portrait_image_path:
            return None
        try:
            return self._resolve_stored_reference(
                session.portrait_image_path,
                allowed_root=self.debug_root,
                error_code="OCR_PORTRAIT_IMAGE_MISSING",
                message="Ảnh chân dung do OCR tạo không còn tồn tại.",
            )
        except FaceSessionError:
            # Luồng Face sẽ tự phát hiện chân dung lại trên cardImage.
            return None

    def _read_unlocked(self, session_id: str) -> FaceSession:
        normalized_id = self._validate_session_id(session_id)
        path = self.session_dir / f"{normalized_id}.json"
        if not path.is_file():
            raise FaceSessionError(
                "FACE_SESSION_NOT_FOUND",
                "Không tìm thấy phiên OCR dùng cho đối chiếu khuôn mặt.",
                status_code=404,
            )

        try:
            with path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            session = FaceSession(**payload)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise FaceSessionError(
                "FACE_SESSION_CORRUPTED",
                "Dữ liệu phiên OCR bị hỏng; vui lòng quét lại CCCD.",
                status_code=500,
            ) from exc

        if session.version != self.RECORD_VERSION:
            raise FaceSessionError(
                "FACE_SESSION_VERSION_UNSUPPORTED",
                "Phiên OCR được tạo bởi phiên bản không còn được hỗ trợ.",
                status_code=409,
            )
        return session

    def _write_unlocked(self, session: FaceSession) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        destination = self.session_dir / f"{session.session_id}.json"
        temporary = self.session_dir / (
            f".{session.session_id}.{uuid4().hex}.tmp"
        )
        try:
            with temporary.open("x", encoding="utf-8") as file:
                json.dump(
                    asdict(session),
                    file,
                    ensure_ascii=False,
                    indent=2,
                )
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink(missing_ok=True)

    def _mark_expired_unlocked(self, session: FaceSession) -> FaceSession:
        if session.status != "active" or not session.is_expired():
            return session
        expired = replace(
            session,
            status="expired",
            updated_at=_format_utc_datetime(_utc_now()),
        )
        self._write_unlocked(expired)
        return expired

    def _cleanup_old_sessions_unlocked(self, now: datetime) -> None:
        if not self.session_dir.exists():
            return

        cutoff = now - timedelta(seconds=self.expired_retention_seconds)
        for path in self.session_dir.glob("*.json"):
            try:
                with path.open("r", encoding="utf-8") as file:
                    payload = json.load(file)
                expires_at = _parse_utc_datetime(str(payload["expires_at"]))
            except (OSError, KeyError, ValueError, json.JSONDecodeError):
                continue
            if expires_at < cutoff:
                path.unlink(missing_ok=True)

    def _acquire_lease_file_unlocked(
        self,
        session_id: str,
    ) -> tuple[Path, str]:
        """Lease file dùng được cả khi Uvicorn chạy nhiều worker."""

        self.session_dir.mkdir(parents=True, exist_ok=True)
        path = self.session_dir / f".{session_id}.lease"
        token = uuid4().hex
        payload = {
            "token": token,
            "pid": os.getpid(),
            "created_at": _format_utc_datetime(_utc_now()),
        }

        for _ in range(2):
            try:
                with path.open("x", encoding="utf-8") as file:
                    json.dump(payload, file, ensure_ascii=False)
                return path, token
            except FileExistsError:
                try:
                    age_seconds = max(0.0, time.time() - path.stat().st_mtime)
                except OSError:
                    continue
                if age_seconds <= self.lease_timeout_seconds:
                    break
                path.unlink(missing_ok=True)

        raise FaceSessionError(
            "FACE_SESSION_BUSY",
            "Phiên OCR đang được đối chiếu bởi một request khác.",
            status_code=409,
        )

    @staticmethod
    def _release_lease_file_unlocked(path: Path, token: str) -> None:
        try:
            with path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, json.JSONDecodeError):
            return
        if payload.get("token") == token:
            path.unlink(missing_ok=True)

    def _validate_reference_image(
        self,
        value: object,
        *,
        allowed_root: Path,
        field_name: str,
    ) -> Path:
        if not value:
            raise FaceSessionError(
                "OCR_REFERENCE_IMAGE_MISSING",
                f"Kết quả OCR không có {field_name} dùng cho Face Verification.",
                status_code=422,
            )

        image_path = Path(str(value))
        if not image_path.is_absolute():
            image_path = self.project_root / image_path
        resolved = image_path.resolve()
        if not resolved.is_relative_to(allowed_root):
            raise FaceSessionError(
                "OCR_REFERENCE_PATH_INVALID",
                "Đường dẫn ảnh tham chiếu của OCR không hợp lệ.",
                status_code=422,
            )
        if not resolved.is_file():
            raise FaceSessionError(
                "OCR_REFERENCE_IMAGE_MISSING",
                "Ảnh tham chiếu do OCR tạo không còn tồn tại.",
                status_code=410,
            )
        return resolved

    def _resolve_stored_reference(
        self,
        value: str,
        *,
        allowed_root: Path,
        error_code: str,
        message: str,
    ) -> Path:
        path = self._absolute_path(value)
        if not path.is_relative_to(allowed_root) or not path.is_file():
            raise FaceSessionError(
                error_code,
                message,
                status_code=410,
            )
        return path

    def _absolute_path(self, value: str | Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self.project_root / path
        return path.resolve()

    def _to_project_relative(self, path: Path) -> str:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.project_root):
            raise FaceSessionError(
                "OCR_REFERENCE_PATH_INVALID",
                "Ảnh OCR nằm ngoài thư mục dự án.",
                status_code=422,
            )
        return resolved.relative_to(self.project_root).as_posix()

    @staticmethod
    def _validate_session_id(session_id: str) -> str:
        value = str(session_id or "").strip()
        if not SESSION_ID_PATTERN.fullmatch(value):
            raise FaceSessionError(
                "INVALID_FACE_SESSION_ID",
                "face session id không hợp lệ.",
                status_code=400,
            )
        return value

    @staticmethod
    def _validate_ocr_request_id(request_id: str) -> str:
        value = str(request_id or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", value):
            raise FaceSessionError(
                "INVALID_OCR_REQUEST_ID",
                "OCR request id không hợp lệ.",
                status_code=422,
            )
        return value


face_session_store = FaceSessionStore()


__all__ = [
    "CaptureSource",
    "FaceSession",
    "FaceSessionError",
    "FaceSessionStore",
    "face_session_store",
]
