"""
IMU Fusion Module
Fuses dual BNO085 IMU quaternion streams (tip + base) into canvas position and pressure.
Extracted and adapted from air_canvas_2.py for OpenCV integration.
"""

import math
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────
#  QUATERNION UTILITIES
# ─────────────────────────────────────────────────────────────────────

def quat_to_euler(w: float, x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert quaternion (w,x,y,z) to Euler angles (roll, pitch, yaw) in radians."""
    # Roll (x-axis)
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)

    # Pitch (y-axis)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)

    return roll, pitch, yaw


def quat_slerp(a: Dict, b: Dict, t: float) -> Dict:
    """Spherical linear interpolation between two quaternion dicts."""
    aw, ax, ay, az = a["w"], a["x"], a["y"], a["z"]
    bw, bx, by, bz = b["w"], b["x"], b["y"], b["z"]

    dot = aw * bw + ax * bx + ay * by + az * bz
    if dot < 0:
        bw, bx, by, bz = -bw, -bx, -by, -bz
        dot = -dot

    if dot > 0.9995:
        rw = aw + t * (bw - aw)
        rx = ax + t * (bx - ax)
        ry = ay + t * (by - ay)
        rz = az + t * (bz - az)
    else:
        theta = math.acos(dot)
        sin_t = math.sin(theta)
        wa = math.sin((1 - t) * theta) / sin_t
        wb = math.sin(t * theta) / sin_t
        rw = wa * aw + wb * bw
        rx = wa * ax + wb * bx
        ry = wa * ay + wb * by
        rz = wa * az + wb * bz

    mag = math.sqrt(rw * rw + rx * rx + ry * ry + rz * rz)
    if mag < 1e-10:
        return {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
    return {"w": rw / mag, "x": rx / mag, "y": ry / mag, "z": rz / mag}


def quat_relative(q_base: Dict, q_tip: Dict) -> Dict:
    """
    Compute the relative rotation from base to tip:
        q_rel = conjugate(q_base) * q_tip
    This isolates the wand's 'bend' or angular difference.
    """
    # conjugate of base
    cw, cx, cy, cz = q_base["w"], -q_base["x"], -q_base["y"], -q_base["z"]
    tw, tx, ty, tz = q_tip["w"], q_tip["x"], q_tip["y"], q_tip["z"]

    return {
        "w": cw * tw - cx * tx - cy * ty - cz * tz,
        "x": cw * tx + cx * tw + cy * tz - cz * ty,
        "y": cw * ty - cx * tz + cy * tw + cz * tx,
        "z": cw * tz + cx * ty - cy * tx + cz * tw,
    }


# ─────────────────────────────────────────────────────────────────────
#  DUAL-IMU FUSION ENGINE
# ─────────────────────────────────────────────────────────────────────

class IMUFusion:
    """
    Fuses tip + base quaternion streams into canvas (x, y, pressure).

    Strategy (quaternion-only):
      • SLERP-average tip & base → "wand midpoint orientation"
      • Extract Euler yaw → canvas X,  pitch → canvas Y
      • Relative rotation (base→tip) roll → pressure/width
      • Subtract calibration pose so the initial hold = canvas center
      • Exponential smoothing on output
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        position_scale: float = 900.0,
        smoothing: float = 0.30,
        dead_zone_rad: float = 0.008,
        pressure_roll_max: float = 0.78,
        draw_threshold: float = 0.12,
        calibration_samples: int = 60,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.position_scale = position_scale
        self.smoothing = smoothing
        self.dead_zone_rad = dead_zone_rad
        self.pressure_roll_max = pressure_roll_max
        self.draw_threshold = draw_threshold
        self.calibration_samples = calibration_samples

        self.reset()

    def reset(self):
        """Reset all state; enters calibration phase."""
        self.calibrated = False
        self.cal_samples_tip: List[Dict] = []
        self.cal_samples_base: List[Dict] = []
        self.cal_yaw = 0.0
        self.cal_pitch = 0.0

        self.smooth_x = 0.5    # normalized 0..1
        self.smooth_y = 0.5
        self.last_tip: Optional[Dict] = None
        self.last_base: Optional[Dict] = None
        self.last_ts = 0

    def _average_quats(self, samples: List[Dict]) -> Dict:
        """Component-wise average + renormalize (fine for small variance)."""
        if not samples:
            return {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
        ref = samples[0]
        sw = sx = sy = sz = 0.0
        for q in samples:
            dot = ref["w"]*q["w"] + ref["x"]*q["x"] + ref["y"]*q["y"] + ref["z"]*q["z"]
            sign = 1.0 if dot >= 0 else -1.0
            sw += sign * q["w"]
            sx += sign * q["x"]
            sy += sign * q["y"]
            sz += sign * q["z"]
        n = len(samples)
        sw /= n; sx /= n; sy /= n; sz /= n
        mag = math.sqrt(sw*sw + sx*sx + sy*sy + sz*sz)
        if mag < 1e-9:
            return {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
        return {"w": sw/mag, "x": sx/mag, "y": sy/mag, "z": sz/mag}

    def update_tip(self, quat: Dict, ts: int):
        """Update with new tip quaternion data."""
        self.last_tip = quat
        self.last_ts = ts
        if not self.calibrated:
            self.cal_samples_tip.append(quat)

    def update_base(self, quat: Dict, ts: int):
        """Update with new base quaternion data."""
        self.last_base = quat
        self.last_ts = ts
        if not self.calibrated:
            self.cal_samples_base.append(quat)

    def try_calibrate(self) -> bool:
        """Attempt calibration if enough samples collected."""
        if self.calibrated:
            return True
        n = min(len(self.cal_samples_tip), len(self.cal_samples_base))
        if n < self.calibration_samples:
            return False

        avg_tip = self._average_quats(self.cal_samples_tip)
        avg_base = self._average_quats(self.cal_samples_base)
        avg = quat_slerp(avg_base, avg_tip, 0.5)
        roll, pitch, yaw = quat_to_euler(avg["w"], avg["x"], avg["y"], avg["z"])

        self.cal_yaw = yaw
        self.cal_pitch = pitch
        self.calibrated = True
        print(f"[IMU CALIBRATED] Reference pose: yaw={math.degrees(yaw):.1f}° pitch={math.degrees(pitch):.1f}°")
        return True

    def get_calibration_progress(self) -> Tuple[int, int]:
        """Returns (current_samples, required_samples) for calibration."""
        n = min(len(self.cal_samples_tip), len(self.cal_samples_base))
        return n, self.calibration_samples

    def get_position(self) -> Optional[Dict]:
        """
        Returns {
            "x": int (pixel x),
            "y": int (pixel y),
            "pressure": float (0..1),
            "is_drawing": bool,
            "roll": float, "pitch": float, "yaw": float
        } or None if not calibrated or no data.
        """
        if not self.calibrated:
            return None
        if self.last_tip is None or self.last_base is None:
            return None

        # ── Fuse tip + base via SLERP ──
        q_avg = quat_slerp(self.last_base, self.last_tip, 0.5)
        roll, pitch, yaw = quat_to_euler(q_avg["w"], q_avg["x"], q_avg["y"], q_avg["z"])

        # ── Subtract calibration reference ──
        delta_yaw = yaw - self.cal_yaw
        delta_pitch = pitch - self.cal_pitch

        # Wrap yaw to [-pi, pi]
        delta_yaw = (delta_yaw + math.pi) % (2 * math.pi) - math.pi

        # ── Map to normalized [0..1] canvas coords ──
        # yaw → X:  positive yaw = move right
        # pitch → Y: positive pitch = move down (screen coords)
        scale = self.position_scale
        raw_x = 0.5 - (delta_yaw * scale) / self.frame_width
        raw_y = 0.5 + (delta_pitch * scale) / self.frame_height

        # ── Dead zone ──
        if (abs(raw_x - self.smooth_x) * self.frame_width < 1.0 and
            abs(raw_y - self.smooth_y) * self.frame_height < 1.0):
            raw_x = self.smooth_x
            raw_y = self.smooth_y

        # ── Exponential smoothing ──
        s = self.smoothing
        self.smooth_x = s * self.smooth_x + (1 - s) * raw_x
        self.smooth_y = s * self.smooth_y + (1 - s) * raw_y

        # Clamp to [0, 1]
        nx = max(0.0, min(1.0, self.smooth_x))
        ny = max(0.0, min(1.0, self.smooth_y))

        # Convert to pixel coordinates
        px = int(nx * self.frame_width)
        py = int(ny * self.frame_height)

        # ── Pressure from relative roll (tip vs base) ──
        q_rel = quat_relative(self.last_base, self.last_tip)
        rel_roll, _, _ = quat_to_euler(q_rel["w"], q_rel["x"], q_rel["y"], q_rel["z"])
        tilt = abs(rel_roll)
        pressure = max(0.0, min(1.0, 1.0 - tilt / self.pressure_roll_max))

        # Determine if drawing (pressure above threshold)
        is_drawing = pressure > self.draw_threshold

        return {
            "x": px,
            "y": py,
            "pressure": pressure,
            "is_drawing": is_drawing,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "delta_yaw": delta_yaw,
            "delta_pitch": delta_pitch,
        }
