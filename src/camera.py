import os
import cv2


def open_camera(width=480, height=480, fps=30):
    """Try camera indices 0-3 and return a configured VideoCapture, or None."""
    preferred_backend = cv2.CAP_DSHOW if os.name == "nt" else 0

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
            return cap
        cap.release()

    return None
