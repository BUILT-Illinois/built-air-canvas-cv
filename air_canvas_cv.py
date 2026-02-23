import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import time
import warnings
import os
import urllib.request

# MQTT Imports
try:
    from mqtt_handler import MQTTHandler, format_hand_data
    from config import (MQTT_ENABLED, MQTT_ENDPOINT, MQTT_CERT_PATH,
                        MQTT_KEY_PATH, MQTT_CA_PATH, MQTT_TOPIC_PREFIX,
                        DEVICE_ID, SEND_INTERVAL_MS)
    MQTT_AVAILABLE = True
except ImportError:
    print("MQTT modules not found. Running without MQTT streaming.")
    MQTT_AVAILABLE = False
    MQTT_ENABLED = False
    DEVICE_ID = "default"
    SEND_INTERVAL_MS = 100

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 0, 0)]
color_names = ["RED", "GREEN", "BLUE", "YELLOW", "MAGENTA", "CYAN", "Black"]
colorIndex = 0
points = [[] for _ in range(len(colors))]


def create_ui(width, height):
    ui_height = height // 8
    ui = np.zeros((ui_height, width, 3), dtype=np.uint8)

    for y in range(ui_height):
        color = [int(240 * (1 - y / ui_height))] * 3
        cv2.line(ui, (0, y), (width, y), color, 1)

    button_width = min(50, width // (len(colors) + 2))
    button_step = button_width + 10
    button_x0 = 10
    button_boxes = []

    for i, color in enumerate(colors):
        x = button_x0 + i * button_step
        cx = x + button_width // 2
        cy = ui_height // 2
        r = max(1, button_width // 2 - 5)

        cv2.circle(ui, (cx, cy), r, color, -1)
        cv2.circle(ui, (cx, cy), r, (0, 0, 0), 2)
        cv2.putText(ui, color_names[i][:1], (cx - 5, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        button_boxes.append((x, 0, x + button_width, ui_height))

    clear_box = (width - 100, 10, width - 10, ui_height - 10)
    cv2.rectangle(ui, (clear_box[0], clear_box[1]), (clear_box[2], clear_box[3]), (200, 200, 200), -1)
    cv2.rectangle(ui, (clear_box[0], clear_box[1]), (clear_box[2], clear_box[3]), (0, 0, 0), 2)
    cv2.putText(ui, "CLEAR", (width - 90, ui_height // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return ui, button_boxes, clear_box


def get_hand_landmarker_model():
    model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
    if not os.path.exists(model_path):
        model_url = (
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task"
        )
        print("Downloading hand landmarker model...")
        urllib.request.urlretrieve(model_url, model_path)
    return model_path


def get_gesture_recognizer_model():
    model_path = os.path.join(os.path.dirname(__file__), "gesture_recognizer.task")
    if not os.path.exists(model_path):
        model_url = (
            "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
            "gesture_recognizer/float16/1/gesture_recognizer.task"
        )
        print("Downloading gesture recognizer model...")
        urllib.request.urlretrieve(model_url, model_path)
    return model_path


# Hand landmarker setup
model_path = get_hand_landmarker_model()
hand_landmarker_options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_landmarker_options)

# Gesture recognizer setup
gesture_model_path = get_gesture_recognizer_model()
gesture_recognizer_options = vision.GestureRecognizerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=gesture_model_path),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
gesture_recognizer = vision.GestureRecognizer.create_from_options(gesture_recognizer_options)

# Initialize MQTT
mqtt_handler = None
if MQTT_AVAILABLE and MQTT_ENABLED:
    print("Initializing MQTT connection...")
    mqtt_handler = MQTTHandler(
        endpoint=MQTT_ENDPOINT,
        cert_path=MQTT_CERT_PATH,
        key_path=MQTT_KEY_PATH,
        ca_path=MQTT_CA_PATH,
        client_id=f"HandTracking-{DEVICE_ID}",
        topic_prefix=MQTT_TOPIC_PREFIX
    )
    mqtt_handler.connect()
    print(f"MQTT streaming: {'ENABLED' if mqtt_handler.connected else 'FAILED'}")
else:
    print("MQTT streaming: DISABLED")


def open_camera():
    preferred_backend = cv2.CAP_DSHOW if os.name == "nt" else 0

    for idx in range(4):
        cap_try = cv2.VideoCapture(idx, preferred_backend) if preferred_backend != 0 else cv2.VideoCapture(idx)
        if not cap_try.isOpened():
            cap_try.release()
            continue
        ret, frame = cap_try.read()
        if ret and frame is not None:
            print(f"Using camera index {idx}")
            return cap_try
        cap_try.release()
    return None


cap = open_camera()
if cap is None:
    print("Error: Could not open any camera (tried indices 0..3).")
    raise SystemExit(1)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

ret, frame = cap.read()
if not ret or frame is None:
    print("Error: Opened camera but failed to read a frame.")
    cap.release()
    raise SystemExit(1)

frame = cv2.flip(frame, 1)
frame_height, frame_width = frame.shape[:2]

ui, color_button_boxes, clear_box = create_ui(frame_width, frame_height)
ui_height = ui.shape[0]
canvas = np.full((frame_height, frame_width, 3), 255, dtype=np.uint8)


def get_index_finger_tip(hand_landmarks):
    return (int(hand_landmarks[8].x * frame_width),
            int(hand_landmarks[8].y * frame_height))

def is_only_index_finger_raised(hand_landmarks):
    index_up = hand_landmarks[8].y < hand_landmarks[6].y
    middle_up = hand_landmarks[12].y < hand_landmarks[10].y
    ring_up = hand_landmarks[16].y < hand_landmarks[14].y
    pinky_up = hand_landmarks[20].y < hand_landmarks[18].y
    return index_up and not (middle_up or ring_up or pinky_up)

def in_box(pt, box):
    x, y = pt
    x1, y1, x2, y2 = box
    return (x1 <= x <= x2) and (y1 <= y <= y2)


prev_point = None
min_distance = 5
is_drawing = False
line_thickness = 2
video_start_time = time.time()

running = True
prev_time = time.time()

current_gesture = "None"
gesture_confidence = 0.0

gesture_cooldowns = {}
COOLDOWN_SECS = 1.0

def gesture_on_cooldown(gesture_name, cooldown=COOLDOWN_SECS):
    curr_time = time.time()
    last_time = gesture_cooldowns.get(gesture_name, 0)
    if curr_time - last_time < cooldown:
        return True
    gesture_cooldowns[gesture_name] = curr_time
    return False

last_mqtt_send_time = 0
mqtt_send_interval = SEND_INTERVAL_MS / 1000.0 if MQTT_AVAILABLE and MQTT_ENABLED else 0

while running:
    current_gesture = "None"
    gesture_confidence = 0.0

    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Failed to capture frame.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int((time.time() - video_start_time) * 1000)
    results = gesture_recognizer.recognize_for_video(mp_image, timestamp_ms)

    if results.hand_landmarks:
        for idx, hand_landmarks in enumerate(results.hand_landmarks):
            index_finger_tip = get_index_finger_tip(hand_landmarks)

            # Get gesture for this hand
            if results.gestures and idx < len(results.gestures):
                gesture = results.gestures[idx][0]
                current_gesture = gesture.category_name
                gesture_confidence = gesture.score

            # Map gestures to actions
            # ["None", "Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou"]
            if current_gesture == "Pointing_Up":  # DRAWING
                if index_finger_tip[1] <= ui_height:  # Toolbar area
                    if in_box(index_finger_tip, clear_box):
                        for p in points:
                            p.clear()
                        canvas.fill(255)
                        prev_point = None
                        is_drawing = False
                    else:
                        for i, box in enumerate(color_button_boxes):
                            if in_box(index_finger_tip, box):
                                colorIndex = i
                                break
                else:
                    if not is_drawing:
                        prev_point = index_finger_tip
                        is_drawing = True

                    if prev_point is not None:
                        if np.linalg.norm(np.array(index_finger_tip) - np.array(prev_point)) > min_distance:
                            cv2.line(canvas, prev_point, index_finger_tip, colors[colorIndex], line_thickness)
                            prev_point = index_finger_tip

            elif current_gesture == "Open_Palm":  # CLEAR CANVAS
                if not gesture_on_cooldown("Open_Palm"):
                    canvas.fill(255)
                    for p in points:
                        p.clear()
                    prev_point = None
                    is_drawing = False

            elif current_gesture == "Victory":  # decide later
                pass

            elif current_gesture == "Thumb_Up":  # increase thickness
                if not gesture_on_cooldown("Thumb_Up", cooldown=0.4):
                    line_thickness = min(line_thickness + 1, 10)

            elif current_gesture == "Thumb_Down":  # decrease thickness
                if not gesture_on_cooldown("Thumb_Down", cooldown=0.4):
                    line_thickness = max(line_thickness - 1, 1)

            elif current_gesture == "ILoveYou":  # save drawing
                if not gesture_on_cooldown("ILoveYou", cooldown=1.0):
                    cv2.imwrite("Air_Sketch_drawing.png", canvas)
                    print("Drawing saved as 'Air_Sketch_drawing.png'")

            elif current_gesture == "Closed_Fist":  # lift pen
                prev_point = None
                is_drawing = False

            else:  # no recognized gesture
                prev_point = None
                is_drawing = False

            cv2.circle(frame, index_finger_tip, 5, colors[colorIndex], -1)

        # Send hand data via MQTT (rate-limited)
        current_time = time.time()
        if mqtt_handler and mqtt_handler.connected and (current_time - last_mqtt_send_time) >= mqtt_send_interval:
            try:
                hand_data = format_hand_data(
                    results.hand_landmarks,
                    frame_width,
                    frame_height,
                    colorIndex,
                    color_names,
                    is_drawing
                )
                mqtt_handler.publish(hand_data)
            except Exception as e:
                print(f"Error sending MQTT data: {e}")
            last_mqtt_send_time = current_time

    else:  # No hand detected
        prev_point = None
        is_drawing = False

    output = frame.copy()
    draw_mask = np.any(canvas != 255, axis=2)
    output[draw_mask] = canvas[draw_mask]
    output[:ui_height, :] = ui

    # Gesture label
    cv2.putText(output, f"Gesture: {current_gesture} ({gesture_confidence:.2f})",
                (10, frame_height - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # FPS counter
    current_time = time.time()
    dt = current_time - prev_time
    fps = (1 / dt) if dt > 0 else 0
    prev_time = current_time
    cv2.putText(output, f"FPS: {int(fps)}", (10, frame_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # MQTT status
    if mqtt_handler:
        mqtt_status = "MQTT: CONNECTED" if mqtt_handler.connected else "MQTT: DISCONNECTED"
        mqtt_color = (0, 255, 0) if mqtt_handler.connected else (0, 0, 255)
        cv2.putText(output, mqtt_status, (10, frame_height - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, mqtt_color, 2)

    cv2.imshow("AirSketch", output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        running = False
    elif key == ord('s'):
        cv2.imwrite("Air_Sketch_drawing.png", canvas)
        print("Drawing saved as 'Air_Sketch_drawing.png'")
    elif key == ord('+'):
        line_thickness = min(line_thickness + 1, 10)
    elif key == ord('-'):
        line_thickness = max(line_thickness - 1, 1)

# Cleanup
if mqtt_handler:
    mqtt_handler.disconnect()

hand_landmarker.close()
gesture_recognizer.close()
cap.release()
cv2.destroyAllWindows()