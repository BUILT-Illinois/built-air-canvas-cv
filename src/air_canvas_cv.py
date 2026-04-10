import time
import warnings

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
from streaming import load_config, start_streaming, stop_streaming
from ui import COLORS, COLOR_NAMES, create_toolbar, point_in_box

# MQTT imports (optional dependency)
try:
    from mqtt_handler import MQTTHandler, format_hand_data
    from config import (MQTT_ENABLED, MQTT_ENDPOINT, MQTT_CERT_PATH,
                        MQTT_KEY_PATH, MQTT_CA_PATH, MQTT_TOPIC_PREFIX,
                        DEVICE_ID, SEND_INTERVAL_MS)
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


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------

def init_mqtt():
    """Connect to AWS IoT Core via MQTT if configured."""
    if not (MQTT_AVAILABLE and MQTT_ENABLED):
        print("MQTT streaming: DISABLED")
        return None

    print("Initializing MQTT connection...")
    handler = MQTTHandler(
        endpoint=MQTT_ENDPOINT,
        cert_path=MQTT_CERT_PATH,
        key_path=MQTT_KEY_PATH,
        ca_path=MQTT_CA_PATH,
        client_id=f"HandTracking-{DEVICE_ID}",
        topic_prefix=MQTT_TOPIC_PREFIX,
    )
    handler.connect()
    print(f"MQTT streaming: {'ENABLED' if handler.connected else 'FAILED'}")
    return handler


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

    # UI toolbar
    toolbar, color_button_boxes, clear_box = create_toolbar(frame_width, frame_height)
    toolbar_height = toolbar.shape[0]

    # Drawing state
    state = DrawingState(frame_width, frame_height)

    # Gesture recognizer
    gesture_recognizer = create_gesture_recognizer()

    # MQTT
    mqtt_handler = init_mqtt()
    mqtt_send_interval = (SEND_INTERVAL_MS / 1000.0
                          if MQTT_AVAILABLE and MQTT_ENABLED else 0)
    last_mqtt_send_time = 0

    # Streaming
    app_config = load_config()
    streamer = start_streaming(app_config, frame_width, frame_height)

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

            if results.hand_landmarks:
                # Ensure per-hand state exists
                for hand_index in range(len(results.hand_landmarks)):
                    state.ensure_hand(hand_index)

                for idx, hand_landmarks in enumerate(results.hand_landmarks):
                    # Draw hand skeleton on camera feed
                    hand_points = landmarks_to_pixels(
                        hand_landmarks, frame_width, frame_height
                    )
                    if len(hand_points) == 21:
                        for start, end in HAND_CONNECTIONS:
                            cv2.line(frame, hand_points[start],
                                     hand_points[end], (0, 255, 0), 2)
                        for pt in hand_points:
                            cv2.circle(frame, pt, 3, (255, 0, 0), -1)

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
                                for i, box in enumerate(color_button_boxes):
                                    if point_in_box(index_tip, box):
                                        state.color_index = i
                                        break
                        else:
                            state.start_drawing(idx, index_tip)

                    elif hand_gesture == "Open_Palm":
                        if not state.gesture_on_cooldown("Open_Palm"):
                            state.clear_canvas()

                    elif hand_gesture == "Victory":
                        pass

                    elif hand_gesture == "Thumb_Up":
                        if not state.gesture_on_cooldown("Thumb_Up", cooldown=0.4):
                            state.increase_thickness()

                    elif hand_gesture == "Thumb_Down":
                        if not state.gesture_on_cooldown("Thumb_Down", cooldown=0.4):
                            state.decrease_thickness()

                    elif hand_gesture == "ILoveYou":
                        if not state.gesture_on_cooldown("ILoveYou", cooldown=1.0):
                            state.save_drawing()

                    elif hand_gesture == "Closed_Fist":
                        state.stop_drawing(idx)

                    else:
                        state.stop_drawing(idx)

                    cv2.circle(frame, index_tip, 5,
                               COLORS[state.color_index], -1)

                # MQTT publish (rate-limited)
                now = time.time()
                if (mqtt_handler and mqtt_handler.connected
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
                        mqtt_handler.publish(hand_data)
                    except Exception as e:
                        print(f"Error sending MQTT data: {e}")
                    last_mqtt_send_time = now

            else:
                # No hands detected
                current_gesture = "None"
                gesture_confidence = 0.0
                state.reset_all_hands()

            # Compose output frame
            output = frame.copy()
            draw_mask = np.any(state.canvas != 255, axis=2)
            output[draw_mask] = state.canvas[draw_mask]
            output[:toolbar_height, :] = toolbar

            # HUD: gesture
            cv2.putText(output,
                        f"Gesture: {current_gesture} ({gesture_confidence:.2f})",
                        (10, frame_height - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # HUD: FPS
            now = time.time()
            dt = now - prev_time
            fps = (1 / dt) if dt > 0 else 0
            prev_time = now
            cv2.putText(output, f"FPS: {int(fps)}", (10, frame_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # HUD: MQTT status
            if mqtt_handler:
                mqtt_status = ("MQTT: CONNECTED" if mqtt_handler.connected
                               else "MQTT: DISCONNECTED")
                mqtt_color = (0, 255, 0) if mqtt_handler.connected else (0, 0, 255)
                cv2.putText(output, mqtt_status, (10, frame_height - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, mqtt_color, 2)

            cv2.imshow("AirSketch", output)

            # Hand the frame off to the dedicated streaming thread.
            # push_frame() is non-blocking — the thread handles timing.
            if streamer is not None:
                streamer.push_frame(output)

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

    finally:
        # Cleanup
        if mqtt_handler:
            mqtt_handler.disconnect()
        gesture_recognizer.close()
        cap.release()
        cv2.destroyAllWindows()
        stop_streaming(streamer)


if __name__ == "__main__":
    main()
