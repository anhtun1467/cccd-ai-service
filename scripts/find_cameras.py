from __future__ import annotations

import time

import cv2


def test_camera(
    index: int,
    backend: int,
    backend_name: str,
) -> bool:
    """
    Kiểm tra một camera index.

    Returns:
        True nếu camera mở và đọc được hình ảnh.
    """
    camera = cv2.VideoCapture(index, backend)

    try:
        if not camera.isOpened():
            return False

        time.sleep(0.3)

        valid_frame = None

        for _ in range(10):
            success, frame = camera.read()

            if success and frame is not None:
                valid_frame = frame

        if valid_frame is None:
            return False

        height, width = valid_frame.shape[:2]

        fps = camera.get(cv2.CAP_PROP_FPS)

        print(
            f"[OK] index={index} | "
            f"backend={backend_name} | "
            f"resolution={width}x{height} | "
            f"fps={fps:.1f}"
        )

        return True

    finally:
        camera.release()


def find_available_cameras(
    max_index: int = 10,
) -> list[int]:
    """
    Tìm toàn bộ camera có thể sử dụng.
    """
    available_indices: set[int] = set()

    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "MediaFoundation"),
    ]

    for index in range(max_index):
        for backend, backend_name in backends:
            try:
                found = test_camera(
                    index=index,
                    backend=backend,
                    backend_name=backend_name,
                )
            except cv2.error as error:
                print(
                    f"[WARNING] Camera {index}: {error}"
                )
                found = False

            if found:
                available_indices.add(index)
                break

    return sorted(available_indices)


def main() -> None:
    print("=" * 70)
    print("TÌM CAMERA TRÊN WINDOWS")
    print("=" * 70)

    camera_indices = find_available_cameras(
        max_index=10
    )

    print("=" * 70)

    if not camera_indices:
        print("[ERROR] Không tìm thấy camera nào.")
        print("")
        print("Kiểm tra:")
        print("1. Logitech C922 đã cắm vào USB.")
        print("2. Đã đóng Camera, OBS, Zoom và Discord.")
        print("3. Windows cho phép ứng dụng desktop dùng camera.")
        return

    print(
        "Camera index khả dụng: "
        f"{camera_indices}"
    )


if __name__ == "__main__":
    main()
