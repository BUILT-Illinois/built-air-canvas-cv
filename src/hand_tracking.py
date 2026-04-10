import os
import urllib.request

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Hand skeleton connections for drawing the wireframe
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (5, 9), (9, 10), (10, 11), (11, 12),      # middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (0, 17),                                  # palm
]


def _ensure_gesture_model():
    """Download the gesture recognizer model if it doesn't exist locally."""
    model_path = os.path.join(os.path.dirname(__file__), "models", "gesture_recognizer.task")
    if not os.path.exists(model_path):
        model_url = (
            "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
            "gesture_recognizer/float16/1/gesture_recognizer.task"
        )
        print("Downloading gesture recognizer model...")
        urllib.request.urlretrieve(model_url, model_path)
    return model_path


def create_gesture_recognizer():
    """Create and return a MediaPipe GestureRecognizer in VIDEO mode."""
    model_path = _ensure_gesture_model()
    options = vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=6,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.GestureRecognizer.create_from_options(options)


def get_index_finger_tip(hand_landmarks, frame_width, frame_height):
    """Return the pixel coordinates of the index-finger tip."""
    return (int(hand_landmarks[8].x * frame_width),
            int(hand_landmarks[8].y * frame_height))


def landmarks_to_pixels(hand_landmarks, frame_width, frame_height):
    """Convert normalized landmarks to a list of (x, y) pixel tuples."""
    return [(int(lm.x * frame_width), int(lm.y * frame_height))
            for lm in hand_landmarks]
