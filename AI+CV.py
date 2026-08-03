import cv2
import numpy as np

cap = cv2.VideoCapture(0)

cv2.namedWindow("Controls")
cv2.createTrackbar("Sat Thresh", "Controls", 150, 255, lambda x: None)
cv2.createTrackbar("Min Area", "Controls", 150, 5000, lambda x: None)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]

    sat_thresh = cv2.getTrackbarPos("Sat Thresh", "Controls")
    min_area = cv2.getTrackbarPos("Min Area", "Controls")

    # Foreground = colored regions
    _, binary = cv2.threshold(s, sat_thresh, 255, cv2.THRESH_BINARY)

    # Small cleanup
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.erode(binary, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=4
    )

    output = frame.copy()
    count = 0

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        count += 1
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cx, cy = centroids[i]
        cv2.circle(output, (int(cx), int(cy)), 4, (0, 0, 255), -1)

        cv2.putText(
            output,
            f"Blob {count}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    cv2.putText(
        output,
        f"Objects: {count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Binary", binary)
    cv2.imshow("Connected Components", output)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()