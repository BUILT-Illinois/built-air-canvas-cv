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
    """Create the toolbar with only a clear button.

    Returns (toolbar_image, clear_box).
    """
    ui_height = height // 8
    ui = np.zeros((ui_height, width, 3), dtype=np.uint8)

    # Gradient background
    for y in range(ui_height):
        color = [int(240 * (1 - y / ui_height))] * 3
        cv2.line(ui, (0, y), (width, y), color, 1)

    # Clear button
    clear_box = (width - 100, 10, width - 10, ui_height - 10)
    cv2.rectangle(ui, (clear_box[0], clear_box[1]),
                  (clear_box[2], clear_box[3]), (200, 200, 200), -1)
    cv2.rectangle(ui, (clear_box[0], clear_box[1]),
                  (clear_box[2], clear_box[3]), (0, 0, 0), 2)

    # Center the text in the button
    text = "CLEAR"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = clear_box[0] + (clear_box[2] - clear_box[0] - text_size[0]) // 2
    text_y = clear_box[1] + (clear_box[3] - clear_box[1] + text_size[1]) // 2
    cv2.putText(ui, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

    return ui, clear_box


def point_in_box(pt, box):
    """Return True if point (x, y) is inside the rectangle (x1, y1, x2, y2)."""
    x, y = pt
    x1, y1, x2, y2 = box
    return (x1 <= x <= x2) and (y1 <= y <= y2)
