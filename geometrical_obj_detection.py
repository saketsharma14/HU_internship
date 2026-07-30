import numpy as np
import cv2

# path = None

# if path:
#     video = cv2.VideoCapture(path)
# else:
#     video = cv2.VideoCapture(0)


# tracked_objects = {}
# next_id = 1
# distance_threshold = 20

# while True:

#     isTrue, frame = video.read()

#     if not isTrue:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     gray = cv2.GaussianBlur(gray, (5,5), 0)

#     edges = cv2.Canny(gray,80,200)

#     contours, _ = cv2.findContours(
#         edges,
#         cv2.RETR_EXTERNAL,
#         cv2.CHAIN_APPROX_SIMPLE
#     )

#     updated_objects = {}          
#     used_ids = set()              

#     for cnt in contours:

#         area = cv2.contourArea(cnt)

#         if area < 1000:
#             continue

#         x, y, w, h = cv2.boundingRect(cnt)

#         cx = x + w // 2
#         cy = y + h // 2

#         matched_id = None
#         min_distance = float("inf")

#         # Compare with previous frame objects
#         for object_id, (old_x, old_y) in tracked_objects.items():

#             if object_id in used_ids:      
#                 continue

#             distance = np.sqrt(
#                 (cx - old_x) ** 2 +
#                 (cy - old_y) ** 2
#             )

#             if distance < min_distance:
#                 min_distance = distance
#                 matched_id = object_id


#         if matched_id is not None and min_distance < distance_threshold:

#             current_id = matched_id

#         else:

#             current_id = next_id
#             next_id += 1


#         updated_objects[current_id] = (cx, cy)   
#         used_ids.add(current_id)                 

#         cv2.drawContours(frame, [cnt], -1, 255, 2)

#         cv2.putText(
#             frame,
#             f"Object {current_id}",
#             (x, y - 10),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.5,
#             255,
#             2
#         )

#     tracked_objects = updated_objects      

#     cv2.imshow("Contours", frame)

#     if cv2.waitKey(20) & 0xFF == ord('q'):
#         break

# video.release()
# cv2.destroyAllWindows()



img = cv2.imread("colored_objects.jpeg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.namedWindow("Image")
cv2.namedWindow("Trackbars")

def nothing(x):
    pass

cv2.createTrackbar("H Min", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("H Max", "Trackbars", 179, 179, nothing)

cv2.createTrackbar("S Min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("S Max", "Trackbars", 255, 255, nothing)

cv2.createTrackbar("V Min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("V Max", "Trackbars", 255, 255, nothing)

while True:

    h_min = cv2.getTrackbarPos("H Min", "Trackbars")
    h_max = cv2.getTrackbarPos("H Max", "Trackbars")
    s_min = cv2.getTrackbarPos("S Min", "Trackbars")
    s_max = cv2.getTrackbarPos("S Max", "Trackbars")
    v_min = cv2.getTrackbarPos("V Min", "Trackbars")
    v_max = cv2.getTrackbarPos("V Max", "Trackbars")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(hsv, lower, upper)

    cv2.imshow("Image", img)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
