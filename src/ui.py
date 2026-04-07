import cv2
import numpy as np

# Color palette shared across the application
COLORS = [
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Cyan
    (0, 0, 0),      # Black
]

COLOR_NAMES = ["RED", "GREEN", "BLUE", "YELLOW", "MAGENTA", "CYAN", "Black"]


def create_toolbar(width, height):
    """Create the color-picker toolbar and clear button.

    Returns (toolbar_image, color_button_boxes, clear_box).
    """
    ui_height = height // 8
    ui = np.zeros((ui_height, width, 3), dtype=np.uint8)

    # Gradient background
    for y in range(ui_height):
        color = [int(240 * (1 - y / ui_height))] * 3
        cv2.line(ui, (0, y), (width, y), color, 1)

    button_width = min(50, width // (len(COLORS) + 2))
    button_step = button_width + 10
    button_x0 = 10
    button_boxes = []

    for i, color in enumerate(COLORS):
        x = button_x0 + i * button_step
        cx = x + button_width // 2
        cy = ui_height // 2
        r = max(1, button_width // 2 - 5)

        cv2.circle(ui, (cx, cy), r, color, -1)
        cv2.circle(ui, (cx, cy), r, (0, 0, 0), 2)
        cv2.putText(ui, COLOR_NAMES[i][:1], (cx - 5, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        button_boxes.append((x, 0, x + button_width, ui_height))

    clear_box = (width - 100, 10, width - 10, ui_height - 10)
    cv2.rectangle(ui, (clear_box[0], clear_box[1]),
                  (clear_box[2], clear_box[3]), (200, 200, 200), -1)
    cv2.rectangle(ui, (clear_box[0], clear_box[1]),
                  (clear_box[2], clear_box[3]), (0, 0, 0), 2)
    cv2.putText(ui, "CLEAR", (width - 90, ui_height // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return ui, button_boxes, clear_box


def point_in_box(pt, box):
    """Return True if point (x, y) is inside the rectangle (x1, y1, x2, y2)."""
    x, y = pt
    x1, y1, x2, y2 = box
    return (x1 <= x <= x2) and (y1 <= y <= y2)
