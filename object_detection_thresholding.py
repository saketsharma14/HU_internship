import cv2

video = cv2.VideoCapture(0)

def nothing():
    pass

cv2.namedWindow("trackbars")

cv2.createTrackbar("H Lower", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("S Lower", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("V Lower", "Trackbars", 0, 255, nothing)

cv2.createTrackbar("H Upper", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("S Upper", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("V Upper", "Trackbars", 255, 255, nothing)


while True:
    ret, frame = video.read()

    if not ret:
        break

    cv2.imshow("Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


# img = cv2.imread("wallet.jpg")
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# cv2.namedWindow("Image")
# cv2.namedWindow("Trackbars")

# def nothing(x):
#     pass

# cv2.createTrackbar("H Min", "Trackbars", 0, 179, nothing)
# cv2.createTrackbar("H Max", "Trackbars", 179, 179, nothing)

# cv2.createTrackbar("S Min", "Trackbars", 0, 255, nothing)
# cv2.createTrackbar("S Max", "Trackbars", 255, 255, nothing)

# cv2.createTrackbar("V Min", "Trackbars", 0, 255, nothing)
# cv2.createTrackbar("V Max", "Trackbars", 255, 255, nothing)

# while True:

#     h_min = cv2.getTrackbarPos("H Min", "Trackbars")
#     h_max = cv2.getTrackbarPos("H Max", "Trackbars")
#     s_min = cv2.getTrackbarPos("S Min", "Trackbars")
#     s_max = cv2.getTrackbarPos("S Max", "Trackbars")
#     v_min = cv2.getTrackbarPos("V Min", "Trackbars")
#     v_max = cv2.getTrackbarPos("V Max", "Trackbars")

#     lower = np.array([h_min, s_min, v_min])
#     upper = np.array([h_max, s_max, v_max])

#     mask = cv2.inRange(hsv, lower, upper)

#     cv2.imshow("Image", img)
#     cv2.imshow("Mask", mask)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
