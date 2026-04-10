import time
from datetime import datetime

import cv2
import numpy as np

from ui import COLORS, COLOR_NAMES


class DrawingState:
    """Manages canvas, color selection, line thickness, and per-hand/IMU draw state."""

    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.canvas = np.full((frame_height, frame_width, 3), 255, dtype=np.uint8)
        self.color_index = 0
        self.line_thickness = 2
        self.min_distance = 5

        # Per-hand tracking
        self.prev_point = {}
        self.is_drawing = {}

        # IMU wand tracking
        self.imu_prev_point = None
        self.imu_is_drawing = False

        # Points history (one list per color)
        self.points = [[] for _ in range(len(COLORS))]

        # Gesture cooldowns
        self._gesture_cooldowns = {}

    # ------------------------------------------------------------------
    # Canvas operations
    # ------------------------------------------------------------------

    def clear_canvas(self):
        self.canvas.fill(255)
        for p in self.points:
            p.clear()
        for hand_idx in list(self.prev_point):
            self.prev_point[hand_idx] = None
            self.is_drawing[hand_idx] = False

    def save_drawing(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Air_Sketch_drawing_{timestamp}.png"
        cv2.imwrite(filename, self.canvas)
        print(f"Drawing saved as '{filename}'")

    def save_drawing_default(self):
        cv2.imwrite("Air_Sketch_drawing.png", self.canvas)
        print("Drawing saved as 'Air_Sketch_drawing.png'")

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def ensure_hand(self, hand_index):
        """Initialize state for a newly detected hand."""
        if hand_index not in self.prev_point:
            self.prev_point[hand_index] = None
            self.is_drawing[hand_index] = False

    def start_drawing(self, hand_index, point):
        if not self.is_drawing[hand_index]:
            self.prev_point[hand_index] = point
            self.is_drawing[hand_index] = True

        if self.prev_point[hand_index] is not None:
            dist = np.linalg.norm(
                np.array(point) - np.array(self.prev_point[hand_index])
            )
            if dist > self.min_distance:
                cv2.line(self.canvas, self.prev_point[hand_index], point,
                         COLORS[self.color_index], self.line_thickness)
                self.prev_point[hand_index] = point

    def stop_drawing(self, hand_index):
        self.prev_point[hand_index] = None
        self.is_drawing[hand_index] = False

    def reset_all_hands(self):
        for hand_idx in list(self.prev_point):
            self.prev_point[hand_idx] = None
            self.is_drawing[hand_idx] = False

    # ------------------------------------------------------------------
    # Thickness
    # ------------------------------------------------------------------

    def increase_thickness(self):
        self.line_thickness = min(self.line_thickness + 1, 10)

    def decrease_thickness(self):
        self.line_thickness = max(self.line_thickness - 1, 1)

    # ------------------------------------------------------------------
    # Gesture cooldown
    # ------------------------------------------------------------------

    def gesture_on_cooldown(self, gesture_name, cooldown=1.0):
        now = time.time()
        last = self._gesture_cooldowns.get(gesture_name, 0)
        if now - last < cooldown:
            return True
        self._gesture_cooldowns[gesture_name] = now
        return False

    # ------------------------------------------------------------------
    # IMU wand drawing
    # ------------------------------------------------------------------

    def draw_imu_wand(self, point, pressure):
        """
        Draw with IMU wand using pressure-sensitive strokes.
        point: (x, y) tuple
        pressure: float 0..1
        """
        if not self.imu_is_drawing:
            self.imu_prev_point = point
            self.imu_is_drawing = True
            return

        if self.imu_prev_point is not None:
            # Check minimum distance
            dist = np.linalg.norm(np.array(point) - np.array(self.imu_prev_point))
            if dist > self.min_distance:
                # Pressure-sensitive thickness (0.3x to 1.0x base thickness)
                pressure_factor = 0.3 + pressure * 0.7
                thickness = max(1, int(self.line_thickness * pressure_factor))

                cv2.line(
                    self.canvas,
                    self.imu_prev_point,
                    point,
                    COLORS[self.color_index],
                    thickness
                )
                self.imu_prev_point = point

    def stop_imu_drawing(self):
        """Stop IMU wand drawing."""
        self.imu_prev_point = None
        self.imu_is_drawing = False
