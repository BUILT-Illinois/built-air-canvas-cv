# Configuration for Air Canvas Combined System
# Supports both hand tracking (MediaPipe) and IMU wand (BNO085)

# ─────────────────────────────────────────────────────────────────────
# Device Identification
# ─────────────────────────────────────────────────────────────────────
DEVICE_ID = "laptop_01"  # Change for each device, to know which device is drawing

# ─────────────────────────────────────────────────────────────────────
# MQTT Configuration (AWS IoT Core)
# ─────────────────────────────────────────────────────────────────────
MQTT_ENABLED = True  # Toggle MQTT streaming on/off
MQTT_ENDPOINT = "aevqdnds5bghe-ats.iot.us-east-1.amazonaws.com"
MQTT_CERT_PATH = "certs/device-certificate.pem.crt"
MQTT_KEY_PATH = "certs/device-private.pem.key"
MQTT_CA_PATH = "certs/AmazonRootCA1.pem"

# Hand Tracking Topics (publisher)
MQTT_TOPIC_PREFIX = f"handtracking/{DEVICE_ID}"
SEND_INTERVAL_MS = 100  # Send data every 100ms (10 times/second)

# IMU Wand Topics (subscriber)
MQTT_TOPIC_IMU_TIP = "air-canvas/imu/tip"
MQTT_TOPIC_IMU_BASE = "air-canvas/imu/base"

# ─────────────────────────────────────────────────────────────────────
# IMU Fusion Parameters
# ─────────────────────────────────────────────────────────────────────
IMU_POSITION_SCALE = 900.0      # Euler radians → pixels conversion
IMU_SMOOTHING = 0.30            # Exponential smoothing (0 = none, 0.95 = max)
IMU_DEAD_ZONE_RAD = 0.008       # Ignore orientation changes below this
IMU_PRESSURE_ROLL_MAX = 0.78    # Roll angle (rad) for zero pressure (~45°)
IMU_DRAW_THRESHOLD = 0.12       # Minimum pressure to draw
IMU_CALIBRATION_SAMPLES = 60    # Samples needed for calibration (~2 sec at 30 Hz)
