"""
Air Canvas Combined Viewer
Combines hand tracking (MediaPipe) and IMU wand (BNO085) on a unified canvas.

Features:
- Hand gesture drawing via webcam
- IMU wand pressure-sensitive drawing
- Dual MQTT streams (hand publisher + IMU subscriber)
- Unified OpenCV display
"""

import time
import warnings
import math

import cv2
import mediapipe as mp
import numpy as np

from camera import open_camera
from drawing import DrawingState
from hand_tracking import (
    HAND_CONNECTIONS,
    create_gesture_recognizer,
    get_index_finger_tip,
    landmarks_to_pixels,
)
from streaming import load_config, start_streaming, stop_streaming, write_frame
from ui import COLORS, COLOR_NAMES, WEB_COLOR_MAP, WEB_BRUSH_SIZE_MAP, create_toolbar, point_in_box
from imu_fusion import IMUFusion

# MQTT imports
try:
    from mqtt_handler import MQTTHandler, MQTTSubscriber, format_hand_data
    from config import (
        MQTT_ENABLED, MQTT_ENDPOINT, MQTT_CERT_PATH,
        MQTT_KEY_PATH, MQTT_CA_PATH, MQTT_TOPIC_CV, MQTT_TOPIC_WAND,
        MQTT_TOPIC_REQUESTS,
        DEVICE_ID, SEND_INTERVAL_MS,
        IMU_POSITION_SCALE, IMU_SMOOTHING, IMU_DEAD_ZONE_RAD,
        IMU_PRESSURE_ROLL_MAX, IMU_DRAW_THRESHOLD, IMU_CALIBRATION_SAMPLES,
    )
    MQTT_AVAILABLE = True
except ImportError as e:
    print(f"MQTT import failed: {e}")
    print("MQTT modules not found. Running without MQTT streaming.")
    MQTT_AVAILABLE = False
    MQTT_ENABLED = False
    DEVICE_ID = "default"
    SEND_INTERVAL_MS = 100

warnings.filterwarnings(
    "ignore", category=UserWarning, module="google.protobuf.symbol_database"
)

WINDOW_MAIN = "Air Canvas Combined"
WINDOW_SKELETON = "Hand Skeleton"


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------

def init_mqtt_publisher():
    """Connect to AWS IoT Core for publishing CV data."""
    if not (MQTT_AVAILABLE and MQTT_ENABLED):
        print("MQTT publishing: DISABLED")
        return None

    print("Initializing MQTT publisher (CV data)...")
    handler = MQTTHandler(
        endpoint=MQTT_ENDPOINT,
        cert_path=MQTT_CERT_PATH,
        key_path=MQTT_KEY_PATH,
        ca_path=MQTT_CA_PATH,
        client_id=f"CV-{DEVICE_ID}",
        topic=MQTT_TOPIC_CV,
    )
    handler.connect()
    print(f"MQTT publisher: {'ENABLED' if handler.connected else 'FAILED'}")
    return handler


def init_mqtt_subscriber(fusion):
    """Connect to AWS IoT Core for subscribing to wand data."""
    if not (MQTT_AVAILABLE and MQTT_ENABLED):
        print("MQTT subscriber: DISABLED")
        return None

    print("Initializing MQTT subscriber (wand data)...")
    subscriber = MQTTSubscriber(
        endpoint=MQTT_ENDPOINT,
        cert_path=MQTT_CERT_PATH,
        key_path=MQTT_KEY_PATH,
        ca_path=MQTT_CA_PATH,
        client_id=f"AirCanvasCombined-{DEVICE_ID}",
    )
    subscriber.connect()

    if subscriber.connected:
        # Subscribe to unified wand topic
        def on_wand_data(topic, msg):
            try:
                ts = msg.get("timestamp", 0)
                data = msg.get("data", {})

                # Extract tip and base quaternions from combined message
                tip_quat = data.get("tip", {}).get("quaternion")
                base_quat = data.get("base", {}).get("quaternion")

                if tip_quat:
                    fusion.update_tip(tip_quat, ts)
                if base_quat:
                    fusion.update_base(base_quat, ts)

            except KeyError as e:
                print(f"[WAND] Missing field: {e}")
            except Exception as e:
                print(f"[WAND] Error processing message: {e}")

        subscriber.subscribe(MQTT_TOPIC_WAND, on_wand_data)

    print(f"MQTT subscriber: {'ENABLED' if subscriber.connected else 'FAILED'}")
    return subscriber


def init_brush_request_subscriber(state, existing_subscriber=None):
    """Subscribe to web UI brush requests (color/size changes)."""
    if not (MQTT_AVAILABLE and MQTT_ENABLED):
        print("[BRUSH] MQTT not available, skipping brush requests")
        return None

    # Reuse existing subscriber or create a new one
    subscriber = existing_subscriber
    if not subscriber or not subscriber.connected:
        print("[BRUSH] Creating dedicated subscriber for brush requests...")
        subscriber = MQTTSubscriber(
            endpoint=MQTT_ENDPOINT,
            cert_path=MQTT_CERT_PATH,
            key_path=MQTT_KEY_PATH,
            ca_path=MQTT_CA_PATH,
            client_id=f"BrushRequests-{DEVICE_ID}",
        )
        subscriber.connect()

    if not subscriber.connected:
        print("[BRUSH] Failed to connect subscriber")
        return None

    def on_brush_request(topic, msg):
        try:
            print(f"[BRUSH] Raw message: {msg}")

            # Handle wrapped envelope (e.g. {"data": {"color":...}}) or flat payload
            data = msg.get("data", msg) if isinstance(msg, dict) else msg

            color = data.get("color", "").lower()
            brush_size = data.get("brushSize", "").lower()

            if color in WEB_COLOR_MAP:
                state.color_index = WEB_COLOR_MAP[color]
                print(f"[BRUSH] Color → {color}")

            if brush_size in WEB_BRUSH_SIZE_MAP:
                state.line_thickness = WEB_BRUSH_SIZE_MAP[brush_size]
                print(f"[BRUSH] Size → {brush_size} (thickness={state.line_thickness})")

        except Exception as e:
            print(f"[BRUSH] Error processing request: {e}")

    subscriber.subscribe(MQTT_TOPIC_REQUESTS, on_brush_request)
    print(f"[BRUSH] Listening for brush requests on {MQTT_TOPIC_REQUESTS}")
    return subscriber


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------

def main():
    # Camera
    cap = open_camera()
    if cap is None:
        print("Error: Could not open any camera (tried indices 0..3).")
        raise SystemExit(1)

    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Opened camera but failed to read a frame.")
        cap.release()
        raise SystemExit(1)

    frame = cv2.flip(frame, 1)
    frame_height, frame_width = frame.shape[:2]

    # Create windows explicitly (macOS can otherwise stack/hide one window)
    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_SKELETON, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_MAIN, frame_width, frame_height)
    cv2.resizeWindow(WINDOW_SKELETON, frame_width, frame_height)

    # Position windows on screen (adjust manually as needed)
    main_x, main_y = 20, 0
    cv2.moveWindow(WINDOW_MAIN, main_x, main_y)
    skeleton_x, skeleton_y = main_x + frame_width // 3, main_y + frame_height // 4
    cv2.moveWindow(WINDOW_SKELETON, skeleton_x, skeleton_y)

    # UI toolbar
    toolbar, clear_box = create_toolbar(frame_width, frame_height)
    toolbar_height = toolbar.shape[0]

    # Drawing state
    state = DrawingState(frame_width, frame_height)

    # Gesture recognizer
    gesture_recognizer = create_gesture_recognizer()

    # IMU fusion
    imu_fusion = IMUFusion(
        frame_width=frame_width,
        frame_height=frame_height,
        position_scale=IMU_POSITION_SCALE if MQTT_AVAILABLE else 900.0,
        smoothing=IMU_SMOOTHING if MQTT_AVAILABLE else 0.30,
        dead_zone_rad=IMU_DEAD_ZONE_RAD if MQTT_AVAILABLE else 0.008,
        pressure_roll_max=IMU_PRESSURE_ROLL_MAX if MQTT_AVAILABLE else 0.78,
        draw_threshold=IMU_DRAW_THRESHOLD if MQTT_AVAILABLE else 0.12,
        calibration_samples=IMU_CALIBRATION_SAMPLES if MQTT_AVAILABLE else 60,
    )

    # MQTT publisher (hand tracking)
    mqtt_publisher = init_mqtt_publisher()
    mqtt_send_interval = (SEND_INTERVAL_MS / 1000.0
                          if MQTT_AVAILABLE and MQTT_ENABLED else 0)
    last_mqtt_send_time = 0

    # MQTT subscriber (IMU wand)
    mqtt_subscriber = init_mqtt_subscriber(imu_fusion)

    # MQTT subscriber (brush requests from website)
    brush_subscriber = init_brush_request_subscriber(state, mqtt_subscriber)

    # Streaming
    app_config = load_config()
    pipe = start_streaming(app_config, frame_width, frame_height)

    # Timing
    video_start_time = time.time()
    prev_time = time.time()

    # HUD state
    current_gesture = "None"
    gesture_confidence = 0.0

    running = True

    try:
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

            # Reset HUD each frame
            current_gesture = "None"
            gesture_confidence = 0.0

            # Skeleton window: separate copy with black background
            skeleton_frame = np.zeros_like(frame)

            # ────────────────────────────────────────────────────────────
            # HAND TRACKING PROCESSING
            # ────────────────────────────────────────────────────────────
            if results.hand_landmarks:
                # Ensure per-hand state exists
                for hand_index in range(len(results.hand_landmarks)):
                    state.ensure_hand(hand_index)

                for idx, hand_landmarks in enumerate(results.hand_landmarks):
                    # Draw hand skeleton on the separate skeleton frame only
                    hand_points = landmarks_to_pixels(
                        hand_landmarks, frame_width, frame_height
                    )
                    if len(hand_points) == 21:
                        for start, end in HAND_CONNECTIONS:
                            cv2.line(skeleton_frame, hand_points[start],
                                     hand_points[end], (0, 255, 0), 2)
                        for pt in hand_points:
                            cv2.circle(skeleton_frame, pt, 3, (255, 0, 0), -1)

                    index_tip = get_index_finger_tip(
                        hand_landmarks, frame_width, frame_height
                    )

                    # Per-hand gesture
                    hand_gesture = "None"
                    hand_confidence = 0.0
                    if results.gestures and idx < len(results.gestures):
                        gesture = results.gestures[idx][0]
                        hand_gesture = gesture.category_name
                        hand_confidence = gesture.score
                        if hand_confidence >= gesture_confidence:
                            current_gesture = hand_gesture
                            gesture_confidence = hand_confidence

                    # --- Gesture actions ---
                    if hand_gesture == "Pointing_Up":
                        if index_tip[1] <= toolbar_height:
                            # Toolbar interaction
                            if point_in_box(index_tip, clear_box):
                                state.clear_canvas()
                        else:
                            state.start_drawing(idx, index_tip)

                    elif hand_gesture == "Open_Palm":
                        if not state.gesture_on_cooldown("Open_Palm"):
                            state.clear_canvas()

                    elif hand_gesture == "Victory":
                        pass

                    elif hand_gesture == "ILoveYou":
                        if not state.gesture_on_cooldown("ILoveYou", cooldown=1.0):
                            state.save_drawing()

                    elif hand_gesture == "Closed_Fist":
                        state.stop_drawing(idx)

                    else:
                        state.stop_drawing(idx)

                    cv2.circle(frame, index_tip, 5,
                               COLORS[state.color_index], -1)

                # MQTT publish hand data (rate-limited)
                now = time.time()
                if (mqtt_publisher and mqtt_publisher.connected
                        and (now - last_mqtt_send_time) >= mqtt_send_interval):
                    try:
                        hand_data = format_hand_data(
                            results.hand_landmarks,
                            frame_width,
                            frame_height,
                            state.color_index,
                            COLOR_NAMES,
                            state.is_drawing,
                        )
                        mqtt_publisher.publish(hand_data)
                    except Exception as e:
                        print(f"Error sending MQTT data: {e}")
                    last_mqtt_send_time = now

            else:
                # No hands detected
                current_gesture = "None"
                gesture_confidence = 0.0
                state.reset_all_hands()

            # ────────────────────────────────────────────────────────────
            # IMU WAND PROCESSING
            # ────────────────────────────────────────────────────────────

            # Try calibration
            imu_fusion.try_calibrate()

            # Get IMU position
            imu_pos = imu_fusion.get_position()
            imu_cursor_x = imu_cursor_y = None
            imu_pressure = 0.0

            if imu_pos:
                imu_cursor_x = imu_pos["x"]
                imu_cursor_y = imu_pos["y"]
                imu_pressure = imu_pos["pressure"]

                # Draw with IMU wand
                if imu_pos["is_drawing"]:
                    state.draw_imu_wand((imu_cursor_x, imu_cursor_y), imu_pressure)
                else:
                    if state.imu_is_drawing:
                        state.stop_imu_drawing()

            # ────────────────────────────────────────────────────────────
            # COMPOSE OUTPUT FRAME
            # ────────────────────────────────────────────────────────────

            output = frame.copy()
            draw_mask = np.any(state.canvas != 255, axis=2)
            output[draw_mask] = state.canvas[draw_mask]
            output[:toolbar_height, :] = toolbar

            # ────────────────────────────────────────────────────────────
            # HUD: Hand gesture
            # ────────────────────────────────────────────────────────────
            cv2.putText(output,
                        f"Hand: {current_gesture} ({gesture_confidence:.2f})",
                        (10, frame_height - 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # HUD: IMU status
            if imu_fusion.calibrated:
                imu_status = f"IMU: Active (P={imu_pressure:.2f})"
                imu_color = (0, 255, 255) if imu_pos and imu_pos["is_drawing"] else (100, 200, 100)
            else:
                cal_progress = imu_fusion.get_calibration_progress()
                imu_status = f"IMU: Calibrating {cal_progress[0]}/{cal_progress[1]}"
                imu_color = (100, 100, 255)

            cv2.putText(output, imu_status, (10, frame_height - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, imu_color, 2)

            # HUD: IMU cursor (if active)
            if imu_cursor_x is not None and imu_cursor_y is not None:
                # Draw crosshair cursor
                cursor_size = int(8 + imu_pressure * 12)
                cursor_color = (0, 245, 255)  # Cyan for IMU
                cv2.circle(output, (imu_cursor_x, imu_cursor_y), cursor_size, cursor_color, 2)
                cv2.line(output, (imu_cursor_x - 10, imu_cursor_y),
                         (imu_cursor_x + 10, imu_cursor_y), cursor_color, 1)
                cv2.line(output, (imu_cursor_x, imu_cursor_y - 10),
                         (imu_cursor_x, imu_cursor_y + 10), cursor_color, 1)

            # HUD: FPS
            now = time.time()
            dt = now - prev_time
            fps = (1 / dt) if dt > 0 else 0
            prev_time = now
            cv2.putText(output, f"FPS: {int(fps)}", (10, frame_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # HUD: MQTT status
            mqtt_status_items = []
            if mqtt_publisher:
                pub_status = "PUB:OK" if mqtt_publisher.connected else "PUB:FAIL"
                mqtt_status_items.append(pub_status)
            if mqtt_subscriber:
                sub_status = f"SUB:OK({mqtt_subscriber.msg_count})" if mqtt_subscriber.connected else "SUB:FAIL"
                mqtt_status_items.append(sub_status)

            if mqtt_status_items:
                mqtt_status_text = " | ".join(mqtt_status_items)
                mqtt_color = (0, 255, 0) if (
                    (not mqtt_publisher or mqtt_publisher.connected) and
                    (not mqtt_subscriber or mqtt_subscriber.connected)
                ) else (0, 0, 255)
                cv2.putText(output, mqtt_status_text, (10, frame_height - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, mqtt_color, 2)

            cv2.imshow(WINDOW_SKELETON, skeleton_frame)
            cv2.imshow(WINDOW_MAIN, output)

            # Stream frame
            pipe = write_frame(pipe, output)

            # Keyboard shortcuts
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                running = False
            elif key == ord("s"):
                state.save_drawing_default()
            elif key == ord("+"):
                state.increase_thickness()
            elif key == ord("-"):
                state.decrease_thickness()
            elif key == ord("r"):
                # Recalibrate IMU
                imu_fusion.reset()
                print("[IMU] Recalibrating... Hold wand still!")

    finally:
        # Cleanup
        if mqtt_publisher:
            mqtt_publisher.disconnect()
        if mqtt_subscriber:
            mqtt_subscriber.disconnect()
        if brush_subscriber and brush_subscriber is not mqtt_subscriber:
            brush_subscriber.disconnect()
        gesture_recognizer.close()
        cap.release()
        cv2.destroyAllWindows()
        stop_streaming(pipe)


if __name__ == "__main__":
    print("=" * 60)
    print("  AIR CANVAS COMBINED — Hand Tracking + IMU Wand")
    print("=" * 60)
    print()
    print("Controls:")
    print("  Hand Gestures:")
    print("    - Pointing Up    : Draw")
    print("    - Open Palm      : Clear canvas")
    print("    - ILoveYou       : Save drawing")
    print()
    print("  Keyboard:")
    print("    - Q   : Quit")
    print("    - S   : Save drawing")
    print("    - R   : Recalibrate IMU wand")
    print()
    print("  Website:")
    print("    - Use website buttons to change color and brush size")
    print()
    print("=" * 60)
    print()
    main()
