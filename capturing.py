import cv2
import os

# Create folder to store images
save_dir = "captured_images"
os.makedirs(save_dir, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

image_count = 0

print("Controls:")
print("  c -> Capture image")
print("  q -> Quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # Show live webcam
    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        filename = os.path.join(save_dir, f"image_{image_count:04d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        image_count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()