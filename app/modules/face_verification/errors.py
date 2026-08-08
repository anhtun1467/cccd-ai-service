from __future__ import annotations

from typing import Any


class FaceVerificationError(RuntimeError):
    """Lỗi nghiệp vụ có mã ổn định của Face Verification."""

    def __init__(
        self,
        error_code: str,
        message: str,
        *,
        status_code: int = 422,
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

