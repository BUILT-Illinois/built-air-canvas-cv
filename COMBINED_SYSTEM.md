# Air Canvas Combined System

## Overview
Unified air drawing system combining **hand tracking** (MediaPipe webcam) and **IMU wand** (dual BNO085 sensors) on a single canvas.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Air Canvas Combined                        │
│                                                               │
│  ┌────────────────────┐         ┌────────────────────┐      │
│  │  Hand Tracking     │         │   IMU Wand         │      │
│  │  (MediaPipe)       │         │   (BNO085)         │      │
│  │                    │         │                    │      │
│  │  • Gesture recog.  │         │  • Tip sensor      │      │
│  │  • Finger drawing  │         │  • Base sensor     │      │
│  │  • Color toolbar   │         │  • Quaternion      │      │
│  └─────────┬──────────┘         └─────────┬──────────┘      │
│            │                               │                  │
│            │ Publishes                     │ Publishes        │
│            │                               │                  │
│            ▼                               ▼                  │
│  ┌──────────────────────────────────────────────────┐        │
│  │          AWS IoT Core (MQTT Broker)              │        │
│  │                                                   │        │
│  │  Topics:                                         │        │
│  │  • handtracking/{device}/data                    │        │
│  │  • air-canvas/imu/tip                            │        │
│  │  • air-canvas/imu/base                           │        │
│  └──────────────────────────────────────────────────┘        │
│            │                               │                  │
│            │ Subscribes                    │                  │
│            ▼                               ▼                  │
│  ┌─────────────────────────────────────────────────┐         │
│  │           Combined Viewer (OpenCV)              │         │
│  │                                                  │         │
│  │  • Unified canvas                                │         │
│  │  • Hand gesture drawing (green cursor)          │         │
│  │  • IMU pressure drawing (cyan cursor)           │         │
│  │  • Real-time fusion & display                   │         │
│  └──────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## Components

### New Modules

1. **`src/imu_fusion.py`**
   - Quaternion utilities (euler conversion, SLERP, relative rotation)
   - `IMUFusion` class - fuses tip+base quaternions → canvas position + pressure
   - Calibration system (60 samples ~2 seconds)
   - Smoothing and dead zone filtering

2. **`src/air_canvas_combined.py`** (MAIN)
   - Combined viewer application
   - Hand tracking processing (existing gestures)
   - IMU wand processing (pressure-sensitive drawing)
   - Dual MQTT connections (publisher + subscriber)
   - Unified OpenCV display

### Enhanced Modules

3. **`src/mqtt_handler.py`**
   - `MQTTHandler` - Publisher (hand tracking data)
   - `MQTTSubscriber` - NEW: Multi-topic subscriber with callbacks

4. **`src/config.py`**
   - Added IMU MQTT topics
   - Added IMU fusion parameters
   - All configuration in one place

5. **`src/drawing.py`**
   - Added IMU wand drawing state
   - `draw_imu_wand()` - pressure-sensitive strokes
   - `stop_imu_drawing()` - reset IMU state

## Running the System

### Prerequisites
```bash
pip install opencv-python mediapipe numpy awsiotsdk pygame
```

### Launch Combined Viewer
```bash
cd src
python air_canvas_combined.py
```

### Controls

**Hand Gestures:**
- **Pointing Up** - Draw with finger
- **Open Palm** - Clear canvas
- **Thumb Up/Down** - Adjust line thickness
- **ILoveYou** (🤟) - Save drawing
- **Closed Fist** - Stop drawing

**Keyboard:**
- `Q` - Quit
- `S` - Save drawing as PNG
- `+/-` - Increase/decrease thickness
- `R` - Recalibrate IMU wand (hold still for 2 seconds)

**IMU Wand:**
- Tilt wand to move cursor (yaw → X, pitch → Y)
- Roll/bend wand for pressure control
- Automatic drawing when pressure > threshold

## Features

### Hand Tracking
- ✅ MediaPipe gesture recognition
- ✅ 6-hand support
- ✅ Color selection via toolbar
- ✅ Publishes to AWS IoT Core
- ✅ Green hand skeleton overlay

### IMU Wand
- ✅ Dual sensor fusion (tip + base)
- ✅ Quaternion-based positioning
- ✅ Pressure-sensitive strokes (roll angle)
- ✅ Auto-calibration on startup
- ✅ Cyan crosshair cursor
- ✅ Smoothing & dead zone filtering

### Combined Features
- ✅ Single unified canvas
- ✅ Draw with BOTH inputs simultaneously
- ✅ Shared color palette
- ✅ Real-time HUD showing:
  - Hand gesture + confidence
  - IMU status + pressure
  - MQTT connection status (PUB/SUB)
  - FPS counter

## Data Flow

### Hand Tracking → AWS
```json
{
  "timestamp": 1234567890,
  "client_id": "HandTracking-laptop_01",
  "data": {
    "landmarks": [{"x": 100.5, "y": 200.3, "z": 0.01}, ...],
    "color": "RED",
    "is_drawing": true
  }
}
```

### IMU Wand → AWS (from Raspberry Pi)
```json
{
  "timestamp": 1234567890,
  "client_id": "RaspberryPi-IMU-01",
  "data": {
    "quaternion": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
    "sample_count": 123
  }
}
```

## Configuration

Edit `src/config.py` to customize:

```python
# MQTT Topics
MQTT_TOPIC_IMU_TIP = "air-canvas/imu/tip"
MQTT_TOPIC_IMU_BASE = "air-canvas/imu/base"

# IMU Fusion Tuning
IMU_POSITION_SCALE = 900.0       # Sensitivity (higher = more movement)
IMU_SMOOTHING = 0.30             # 0 = raw, 0.95 = very smooth
IMU_PRESSURE_ROLL_MAX = 0.78     # Roll angle for zero pressure
IMU_DRAW_THRESHOLD = 0.12        # Minimum pressure to draw
```

## Troubleshooting

### IMU Not Calibrating
- Ensure Raspberry Pi is publishing to correct topics
- Check AWS IoT Core connection
- Hold wand completely still for 2-3 seconds
- Press `R` to restart calibration

### Hand Tracking Issues
- Ensure good lighting
- Check camera permissions
- Verify MediaPipe model download

### MQTT Connection Failed
- Verify certificate paths in `config.py`
- Check AWS IoT Core endpoint
- Ensure certificates have correct permissions

## Next Steps

- [ ] Add gesture to switch between hand/IMU mode
- [ ] Implement drawing layers (toggle visibility)
- [ ] Add color picker gesture for IMU wand
- [ ] Save/load drawing sessions
- [ ] Multi-user collaborative mode
- [ ] 3D drawing mode using IMU depth data

## File Structure
```
src/
├── air_canvas_combined.py    # Main combined viewer ⭐ NEW
├── imu_fusion.py              # IMU quaternion fusion ⭐ NEW
├── mqtt_handler.py            # Publisher + Subscriber (enhanced)
├── config.py                  # Unified configuration (enhanced)
├── drawing.py                 # Drawing state + IMU support (enhanced)
├── hand_tracking.py           # MediaPipe gestures
├── camera.py                  # Camera management
├── ui.py                      # Toolbar & colors
├── streaming.py               # Optional video streaming
└── websocket_handler.py       # Optional websocket
```

## Credits
Built by BUILT @ Illinois for EOH 2026
