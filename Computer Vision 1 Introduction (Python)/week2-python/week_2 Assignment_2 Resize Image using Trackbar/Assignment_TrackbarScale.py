import cv2


def passFuntion(*args):
    pass


def do_scalling(scaleFactor, scaletype):
    if scaletype == 0:
        scaleFactor = 1 + scaleFactor/100.0
    if scaletype == 1:
        scaleFactor = 1 - scaleFactor/100.0
    return scaleFactor


def main():
    # Load image
    img = cv2.imread('truth.png')
    # We will create two trackbars in one windows,
    # so the WindowName should be the same
    windowName = "Resize Image"

    # 1. For getting the percentage of scaling to be done.
    trackbarScale = "Scale"     # Title showed in the Scale trackbar
    maxScaleUp = 100    # maximum value
    currentScale = 0     # increase by

    # 2. Trackbar for getting the scaling type
    trackbarType = "Type: \n 0: Scale Up \n 1: Scale Down"
    maxType = 1     # maximum type (0 is scale UP, 1 is scale Down)
    currentType = 0   # current
    cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)

    cv2.createTrackbar(trackbarScale, windowName, currentScale,
                       maxScaleUp, passFuntion)
    cv2.createTrackbar(trackbarType, windowName, currentType,
                       maxType, passFuntion)

    while True:

        scaleType = cv2.getTrackbarPos(trackbarType, windowName)
        scaleFactor = cv2.getTrackbarPos(trackbarScale, windowName)
        scaleFactor = do_scalling(scaleFactor, scaleType)

        scaledImg = cv2.resize(img, None, fx=scaleFactor, fy=scaleFactor,
                               interpolation=cv2.INTER_LINEAR)
        cv2.imshow(windowName, scaledImg)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


# Initialization
cv2.imshow(windowName, img)

# Create Trackbar to choose percentage of scaling
cv2.createTrackbar(trackbarValue, windowName,
                   scaleFactor, maxScaleUp, scaleImage)

# Create Trackbar to choose type of scaling ( Up or down )
cv2.createTrackbar(trackbarType, windowName,
                   scaleType, maxType, scaleTypeImage)

# Calling the function for the first time
# scaleImage(25)
# Start the loop to record
key = 0
while key != 27:
    key = cv2.waitKey(20) & 0xFF


cv2.destroyAllWindows()
