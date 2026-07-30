import cv2
import numpy as np
from collections import deque

lower = np.array([87, 5, 19])
upper = np.array([128, 200, 152])

MIN_AREA = 500
TRAIL_LENGTH = 100

video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Could not open camera")
    exit()

kernel = np.ones((5, 5), np.uint8)

# Stores the last TRAIL_LENGTH (cx, cy) pixel positions
trail = deque(maxlen=TRAIL_LENGTH)

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    height, width = mask.shape
    center_x = width // 2
    center_y = height // 2

    M = cv2.moments(mask, binaryImage=True)

    display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    cv2.line(display, (center_x, 0), (center_x, height), (0, 255, 0), 1)
    cv2.line(display, (0, center_y), (width, center_y), (0, 255, 0), 1)

    if M["m00"] > MIN_AREA:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        rel_x = cx - center_x
        rel_y = center_y - cy

        # Add current position to the trail
        trail.appendleft((cx, cy))

        cv2.circle(display, (cx, cy), 6, (0, 0, 255), -1)

        label = f"({rel_x}, {rel_y})"
        label_x = min(cx + 10, width - 120)
        label_y = max(cy - 10, 20)
        cv2.putText(display, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cx, cy = None, None
        # Object lost this frame — optionally break the trail here
        # trail.appendleft(None)

    # Draw the trajectory: connect consecutive points, fading with age
    for i in range(1, len(trail)):
        if trail[i - 1] is None or trail[i] is None:
            continue
        # Fade thickness/brightness with age (older = thinner/dimmer)
        thickness = max(1, int(np.sqrt(TRAIL_LENGTH / float(i + 1)) * 2))
        cv2.line(display, trail[i - 1], trail[i], (255, 150, 0), thickness)

    cv2.imshow("Masked Image", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()