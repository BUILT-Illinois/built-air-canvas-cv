import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import time
import warnings
import os
import urllib.request
import json
import subprocess

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

# Load configuration
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    if not os.path.exists(config_path):
        return {"streaming": {"enabled": False}}
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except:
        return {"streaming": {"enabled": False}}

config = load_config()

# Check if ffmpeg is available
def ffmpeg_available():
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

STREAMING_ENABLED = config.get('streaming', {}).get('enabled', False) and ffmpeg_available()

if config.get('streaming', {}).get('enabled', False) and not ffmpeg_available():
    print("Warning: Streaming enabled in config but ffmpeg not found. Streaming disabled.")

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

# Setup streaming if enabled
pipe = None
if STREAMING_ENABLED:
    # Validate required YouTube streaming configuration before starting
    youtube_config = config.get('youtube') if isinstance(config, dict) else None
    if not isinstance(youtube_config, dict) or \
       'stream_url' not in youtube_config or \
       'stream_key' not in youtube_config:
        print("Streaming disabled: missing YouTube 'stream_url' or 'stream_key' in configuration.")
        STREAMING_ENABLED = False
        pipe = None
    else:
        try:
            stream_url = youtube_config['stream_url']
            stream_key = youtube_config['stream_key']
            full_url = f"{stream_url}{stream_key}"

            bitrate = config.get('streaming', {}).get('bitrate', '3000k')
            preset = config.get('streaming', {}).get('preset', 'ultrafast')

            ffmpeg_cmd = [
                'ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{frame_width}x{frame_height}',
                '-r', '30',
                '-i', 'pipe:0',
                '-f', 'lavfi',
                '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', preset,
                '-b:v', bitrate,
                '-maxrate', bitrate,
                '-bufsize', '6000k',
                '-g', '60',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-ar', '44100',
                '-f', 'flv',
                full_url
            ]

            pipe = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                bufsize=10**8
            )

            print("YouTube streaming started!")
            print(f"Streaming to: {stream_url}")
            print(f"Resolution: {frame_width}x{frame_height}")

        except Exception as e:
            print(f"Failed to start streaming: {e}")
            import traceback
            traceback.print_exc()
            pipe = None
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
                if index_finger_tip[1] <= ui_height:
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
            else:
                prev_point = None
                is_drawing = False

            cv2.circle(frame, index_finger_tip, 5, colors[colorIndex], -1)

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

    cv2.imshow("AirSketch", output)
    
    # Stream to YouTube if enabled
    if pipe is not None:
        try:
            pipe.stdin.write(output.tobytes())
        except (BrokenPipeError, IOError) as e:
            print(f"Streaming error: {e}")
            # Clean up the streaming process before dropping the reference
            try:
                if pipe.stdin:
                    pipe.stdin.close()
                try:
                    pipe.terminate()
                except Exception:
                    # If terminate is not available or fails, proceed to kill in wait branch
                    pass
                try:
                    pipe.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pipe.kill()
                    try:
                        pipe.wait()
                    except Exception:
                        pass
            except Exception:
                # Swallow any cleanup errors; we are already handling a failure path
                pass
            finally:
                pipe = None

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
if pipe is not None:
    try:
        pipe.stdin.close()
        pipe.wait(timeout=5)
    except:
        pipe.kill()

cap.release()
hand_landmarker.close()
cv2.destroyAllWindows()
