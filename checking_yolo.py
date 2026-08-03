
import ultralytics
import cv2

model = ultralytics.YOLO("yolo11n-seg.pt")
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    results = model(frame)

    # Draw detections
    annotated_frame = results[0].plot()

    cv2.imshow("live_cam", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()