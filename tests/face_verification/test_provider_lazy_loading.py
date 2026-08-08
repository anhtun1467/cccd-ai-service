from __future__ import annotations

from app.core.face_verification_provider import FaceVerificationProvider
from app.modules.face_verification.embedding import InsightFaceEmbedder


def test_embedder_does_not_load_model_in_constructor() -> None:
    embedder = InsightFaceEmbedder()
    assert embedder.is_loaded is False


def test_provider_builds_service_only_on_first_access(monkeypatch) -> None:
    provider = FaceVerificationProvider()
    sentinel = object()
    build_count = 0

    def fake_build_service() -> None:
        nonlocal build_count
        build_count += 1
        provider._service = sentinel  # type: ignore[assignment]

    monkeypatch.setattr(provider, "_build_service", fake_build_service)

    assert provider._service is None
    assert provider.service is sentinel
    assert provider.service is sentinel
    assert build_count == 1

