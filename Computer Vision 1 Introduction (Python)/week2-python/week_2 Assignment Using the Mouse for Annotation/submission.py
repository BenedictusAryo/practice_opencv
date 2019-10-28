# Week 2 Assignment using Mouse for annotation
# OpenCV Cropping tools GUI using mouse
# Benedictus Aryo

import cv2

# List to store the points
topLeft = []
bottomRight = []


def drawRectangle(action, x, y, flags, userdata):
    # Referencing global variables
    global topLeft, bottomRight
    if action == cv2.EVENT_LBUTTONDOWN:
        topLeft.append((x, y))
        # print("Top left: ", topLeft)
        # Mark the topleft
        cv2.circle(source, topLeft[0], 1,
                   (255, 255, 0), 2, cv2.LINE_AA)

    # When left mouse button is released
    elif action == cv2.EVENT_LBUTTONUP:
        bottomRight.append((x, y))
        # print("Bottom Right: ", bottomRight)
        cv2.rectangle(source, topLeft[0], bottomRight[0], (0, 255, 0), 2)
        face = source[topLeft[0][1]:bottomRight[0]
                      [1], topLeft[0][0]:bottomRight[0][0]]
        # print(face)
        cv2.imwrite('face_output.png', face)
        cv2.imshow("Window", source)
        topLeft.clear()
        bottomRight.clear()


source = cv2.imread('sample.jpg', 1)
print("image shape: ", source.shape)
dummy = source.copy()

cv2.namedWindow('Window')
cv2.setMouseCallback('Window', drawRectangle)
k = 0

while k != 27:
    cv2.putText(source, 'Choose top left corner and drag, Press ESC to exit',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1)
    cv2.imshow("Window", source)
    k = cv2.waitKey(0) & 0xFF
    # Another way of cloning
    if k == 99:
        source = dummy.copy()

cv2.destroyAllWindows()
