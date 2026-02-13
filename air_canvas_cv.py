import cv2
import numpy as np
import mediapipe as mp
import time
import warnings
import os

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


mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1,model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

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
    return (int(hand_landmarks.landmark[8].x * frame_width),
            int(hand_landmarks.landmark[8].y * frame_height))

def is_index_finger_raised(hand_landmarks):
    return hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y

def in_box(pt, box):
    x, y = pt
    x1, y1, x2, y2 = box
    return (x1 <= x <= x2) and (y1 <= y <= y2)

prev_point = None
min_distance = 5
is_drawing = False
line_thickness = 2

running = True
prev_time = time.time()

while running:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Failed to capture frame.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger_tip = get_index_finger_tip(hand_landmarks)

            if is_index_finger_raised(hand_landmarks):
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

    output = cv2.addWeighted(frame, 0.6, canvas, 0.4, 0)
    output[:ui_height, :] = ui

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
cv2.destroyAllWindows()
