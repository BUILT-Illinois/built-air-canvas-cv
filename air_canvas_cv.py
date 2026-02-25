import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import time
import warnings
import os
import urllib.request

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

# Gesture recognition
def get_gesture_recognizer_model():
    model_path = os.path.join(os.path.dirname(__file__), "gesture_recognizer.task")
    if not os.path.exists(model_path):
        model_url = ( # double check
            "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
            "gesture_recognizer/float16/1/gesture_recognizer.task"
        )
        print("Downloading gesture recognizer model...")
        urllib.request.urlretrieve(model_url, model_path)
    return model_path

# gesture recog. setup
gesture_model_path = get_gesture_recognizer_model()
gesture_recognizer_options = vision.GestureRecognizerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=gesture_model_path),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5, # in video mode, if 
    min_tracking_confidence=0.5,
)
gesture_recognizer = vision.GestureRecognizer.create_from_options(gesture_recognizer_options)




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

gesture_confidence = 0.0
current_gesture = "None"

gesture_cooldowns = {}
COOLDOWN_SECS = 1.0

def gesture_on_cooldown(gesture_name, cooldown=COOLDOWN_SECS):
    curr_time = time.time()
    last_time = gesture_cooldowns.get(gesture_name, 0)

    if curr_time - last_time < cooldown:
        return True
    
    gesture_cooldowns[gesture_name] = curr_time
    return False

while running: 
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

            # get gesture for this specific hand
            if results.gestures and idx < len(results.gestures):
                gesture = results.gestures[idx][0]
                current_gesture = gesture.category_name
                gesture_confidence = gesture.score
                # cv2.putText(output, f"Gesture: {gesture_name} ({score:.2f})", (10, frame_height - 40),
                # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2) # yellow
    
            # map gestures to (7) actions ["None", "Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou"].
            if current_gesture == "Pointing_Up": # DRAWING
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
            elif current_gesture == "Open_Palm": # CLEAR CANVAS
                if not gesture_on_cooldown("Open_Palm"):
                    canvas.fill(255)
                    for p in points:
                        p.clear()
                    prev_point = None
                    is_drawing = False
            elif current_gesture == "Victory": # decide later
                pass
            elif current_gesture == "Thumb_Up": # same as '+' for now
                if not gesture_on_cooldown("Thumb_Up", cooldown=0.4):  
                    line_thickness = min(line_thickness + 1, 10)
            elif current_gesture == "Thumb_Down": # same as '-' for now
                if not gesture_on_cooldown("Thumb_Down", cooldown=0.4):
                    line_thickness = max(line_thickness - 1, 1)
            elif current_gesture == "ILoveYou": # same as 's' for now
                if not gesture_on_cooldown("ILoveYou", cooldown=1.0): # adjust if necessary
                    cv2.imwrite("Air_Sketch_drawing.png", canvas)
                    print("Drawing saved as 'Air_Sketch_drawing.png'")
            elif current_gesture == "Closed_Fist":
                prev_point = None
                is_drawing = False
            else: # no gesture
                prev_point = None
                is_drawing = False

            cv2.circle(frame, index_finger_tip, 5, colors[colorIndex], -1)
    else: # no hand detected, reset gesture info
        current_gesture = "None"
        gesture_confidence = 0.0
        prev_point = None
        is_drawing = False


    output = frame.copy()
    draw_mask = np.any(canvas != 255, axis=2)
    output[draw_mask] = canvas[draw_mask]
    output[:ui_height, :] = ui


    # gesture_name = ""
    # if results.gestures:
    #     gesture_name = results.gestures[0][0].category_name
    #     score = results.gestures[0][0].score
    gesture_color = (0, 255, 0) # green if detected
    cv2.putText(output, f"Gesture: {current_gesture} ({gesture_confidence:.2f})",
                (10, frame_height - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, gesture_color, 2) 
    


    current_time = time.time()
    dt = current_time - prev_time
    fps = (1 / dt) if dt > 0 else 0
    prev_time = current_time

    cv2.putText(output, f"FPS: {int(fps)}", (10, frame_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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

cap.release()
hand_landmarker.close()
gesture_recognizer.close()
cv2.destroyAllWindows()