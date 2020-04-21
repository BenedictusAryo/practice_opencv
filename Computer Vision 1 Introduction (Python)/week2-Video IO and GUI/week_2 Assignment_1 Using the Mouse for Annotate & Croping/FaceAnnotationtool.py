import cv2

# List to store the points
topLeft = []
bottomRight = []


def drawRectangle(action, x, y, flags, userdata):
    # Referencing global variables
    global topLeft, bottomRight
    if action == cv2.EVENT_LBUTTONDOWN:
        topLeft = [(x, y)]
        # Mark the topleft
        cv2.circle(source, topLeft[0], 1,
                   (255, 255, 0), 2, cv2.LINE_AA)

    # When left mouse button is released
    elif action == cv2.EVENT_LBUTTONUP:
        bottomRight = [(x, y)]
        cv2.rectangle(source, topLeft[0], bottomRight[0], (0, 255, 0), 2)
        face = source[topLeft[0], bottomRight[0]]
        cv2.imwrite('face.png', face)
        cv2.imshow("Window", source)


source = cv2.imread('sample.jpg', 1)
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
