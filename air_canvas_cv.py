import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import time
import warnings
import os
import urllib.request

#Websocket Imports
try:
    from websocket_handler import WebSocketHandler, format_hand_data
    from config import WEBSOCKET_ENABLED, WEBSOCKET_URL, DEVICE_ID, SEND_INTERVAL_MS
    WEBSOCKET_AVAILABLE = True
except ImportError:
    print("WebSocket modules not found. Running without WebSocket streaming.")
    WEBSOCKET_AVAILABLE = False
    WEBSOCKET_ENABLED = False

#MQTT Imports
try:
    from mqtt_handler import MQTTHandler
    from config import (MQTT_ENABLED, MQTT_ENDPOINT, MQTT_CERT_PATH, 
                        MQTT_KEY_PATH, MQTT_CA_PATH, MQTT_TOPIC_PREFIX)
    MQTT_AVAILABLE = True
except ImportError:
    print("MQTT modules not found. Running without MQTT streaming.")
    MQTT_AVAILABLE = False
    MQTT_ENABLED = False


warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0),(0,0,0)]
color_names = ["RED", "GREEN", "BLUE", "YELLOW", "MAGENTA", "CYAN","Black"]
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
    button_x0 = 10  # first button start
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

        # store a rectangular hitbox around the circular button
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


model_path = get_hand_landmarker_model()
base_options = mp_python.BaseOptions(model_asset_path=model_path)
hand_landmarker_options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_landmarker_options)

#Initialize Websocket
ws_handler = None
if WEBSOCKET_AVAILABLE and WEBSOCKET_ENABLED:
    print("Initializing WebSocket connection...")
    ws_handler = WebSocketHandler(WEBSOCKET_URL, DEVICE_ID)
    ws_handler.connect()
    print(f"WebSocket streaming: {'ENABLED' if ws_handler.connected else 'FAILED'}")
else:
    print("WebSocket streaming: DISABLED")

#Initialize MQTT
mqtt_handler = None
if MQTT_AVAILABLE and MQTT_ENABLED:
    import os
    print("Initializing MQTT connection...")
    mqtt_handler = MQTTHandler(
        endpoint=MQTT_ENDPOINT,
        cert_path=os.path.join(os.path.dirname(__file__), MQTT_CERT_PATH),
        key_path=os.path.join(os.path.dirname(__file__), MQTT_KEY_PATH),
        ca_path=os.path.join(os.path.dirname(__file__), MQTT_CA_PATH),
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

# Determines size of window screen
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Read one frame to lock actual size
ret, frame = cap.read()
if not ret or frame is None:
    print("Error: Opened camera but failed to read a frame.")
    cap.release()
    raise SystemExit(1)

frame = cv2.flip(frame, 1)
frame_height, frame_width = frame.shape[:2]

# Build UI and canvas based on actual size
ui, color_button_boxes, clear_box = create_ui(frame_width, frame_height)    
ui_height = ui.shape[0]
canvas = np.full((frame_height, frame_width, 3), 255, dtype=np.uint8)

def get_index_finger_tip(hand_landmarks):
    return (int(hand_landmarks[8].x * frame_width),
            int(hand_landmarks[8].y * frame_height))

def is_index_finger_raised(hand_landmarks):
    return hand_landmarks[8].y < hand_landmarks[6].y

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

last_ws_send_time = 0
ws_send_interval = SEND_INTERVAL_MS / 1000.0 if WEBSOCKET_AVAILABLE and WEBSOCKET_ENABLED else 0

while running:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Failed to capture frame.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int((time.time() - video_start_time) * 1000)
    results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            index_finger_tip = get_index_finger_tip(hand_landmarks)

            if is_only_index_finger_raised(hand_landmarks):
                if index_finger_tip[1] <= ui_height:  # Toolbar area
                    # Clear button
                    if in_box(index_finger_tip, clear_box):
                        for p in points:
                            p.clear()
                        canvas.fill(255)
                        prev_point = None
                        is_drawing = False
                    else:
                        # Color buttons
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
            else:
                prev_point = None
                is_drawing = False

            cv2.circle(frame, index_finger_tip, 5, colors[colorIndex], -1)

        current_time = time.time()
        if results.hand_landmarks and (current_time - last_ws_send_time) >= ws_send_interval:
            hand_data = format_hand_data(
                results.hand_landmarks,
                frame_width,
                frame_height,
                colorIndex,
                color_names,
                is_drawing
            )
            
            # Send to WebSocket
            if ws_handler and ws_handler.connected:
                try:
                    ws_handler.send_data(hand_data)
                except Exception as e:
                    print(f"Error sending WebSocket data: {e}")

            # Send to MQTT
            if mqtt_handler and mqtt_handler.connected:
                try:
                    mqtt_handler.publish(hand_data)
                except Exception as e:
                    print(f"Error sending MQTT data: {e}")
            
            last_ws_send_time = current_time  # ADD THIS LINE!


    output = frame.copy()
    draw_mask = np.any(canvas != 255, axis=2)
    output[draw_mask] = canvas[draw_mask]
    output[:ui_height, :] = ui

    current_time = time.time()
    dt = current_time - prev_time
    fps = (1 / dt) if dt > 0 else 0
    prev_time = current_time

    cv2.putText(output, f"FPS: {int(fps)}", (10, frame_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    #Websocket status
    if ws_handler:
        ws_status = "WS: CONNECTED" if ws_handler.connected else "WS: DISCONNECTED"
        ws_color = (0, 255, 0) if ws_handler.connected else (0, 0, 255)
        cv2.putText(output, ws_status, (10, frame_height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, ws_color, 2)
        
    #MQTT status
    if mqtt_handler:
        mqtt_status = "MQTT: CONNECTED" if mqtt_handler.connected else "MQTT: DISCONNECTED"
        mqtt_color = (0, 255, 0) if mqtt_handler.connected else (0, 0, 255)
        cv2.putText(output, mqtt_status, (10, frame_height - 70),
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

#Disconnects websocket
if ws_handler:
    ws_handler.disconnect()

#Disconnects MQTT
if mqtt_handler:
    mqtt_handler.disconnect()

hand_landmarker.close()
cap.release()
cv2.destroyAllWindows()