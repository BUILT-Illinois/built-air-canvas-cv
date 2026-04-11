import platform
import cv2


def open_camera(width=1920, height=1080, fps=60):
    """Try camera indices 0-3 and return a configured VideoCapture, or None."""
    system = platform.system()
    if system == "Windows":
        preferred_backend = cv2.CAP_DSHOW
    elif system == "Darwin":
        preferred_backend = cv2.CAP_AVFOUNDATION
    else:
        preferred_backend = 0

    for idx in range(4):
        cap = (cv2.VideoCapture(idx, preferred_backend)
            if preferred_backend != 0
            else cv2.VideoCapture(idx))
        if not cap.isOpened():
            cap.release()
            continue
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"Using camera index {idx}")
            cap.set(cv2.CAP_PROP_FOURCC,
                    cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimal buffer = lowest latency
            return cap
        cap.release()

    return None
