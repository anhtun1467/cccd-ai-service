from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from app.core.face_verification_provider import (
    FaceVerificationProvider,
    provider,
)


def main() -> None:
    provider_again = FaceVerificationProvider.instance()

    print("=" * 64)
    print("FACE VERIFICATION PROVIDER TEST")
    print("=" * 64)

    print(f"Provider ID 1:           {id(provider)}")
    print(f"Provider ID 2:           {id(provider_again)}")
    print(f"Same provider:           {provider is provider_again}")

    print()

    print(f"Service ID 1:            {id(provider.service)}")
    print(f"Service ID 2:            {id(provider_again.service)}")
    print(
        "Same service:            "
        f"{provider.service is provider_again.service}"
    )

    print()

    print(f"Embedder ID 1:           {id(provider.embedder)}")
    print(f"Embedder ID 2:           {id(provider_again.embedder)}")
    print(
        "Same embedder:           "
        f"{provider.embedder is provider_again.embedder}"
    )

    print()

    print(
        "Portrait extractor ID 1: "
        f"{id(provider.portrait_extractor)}"
    )
    print(
        "Portrait extractor ID 2: "
        f"{id(provider_again.portrait_extractor)}"
    )
    print(
        "Same portrait extractor: "
        f"{provider.portrait_extractor is provider_again.portrait_extractor}"
    )

    print("=" * 64)


if __name__ == "__main__":
    main()
