import cv2

# We will create two trackbars in one windows,
# so the WindowName should be the same
windowName = "Resize Image"

# 1. For getting the percentage of scaling to be done.
maxScaleUp = 100    # maximum value
scaleFactor = 1     # increase by
trackbarValue = "Scale"     # Title showed in the Scale trackbar

# 2. Trackbar for getting the scaling type
maxType = 1     # maximum type (0 is scale UP, 1 is scale Down)
scaleType = 0   # current
trackbarType = "Type: \n 0: Scale Up \n 1: Scale Down"


# Load image
img = cv2.imread('truth.png')

# Create a window to display results
cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)


# Create Callback Function that scale the image
def scaleImage(*args):
    global scaleFactor
    global scaleType

    # Get the scale factor from the trackbar
    scaleFactor = 1 + args[0]/100.0

    # Perform check if scaleFactor is zero
    if scaleFactor == 0:
        scaleFactor = 1

    # Resize the image
    scaledImage = cv2.resize(img, None, fx=scaleFactor,
                             fy=scaleFactor,
                             interpolation=cv2.INTER_LINEAR)
    cv2.imshow(windowName, scaledImage)


# Create Callback Function that change the Type Scale
def scaleTypeImage(*args):
    # Referencing global variables
    global scaleType
    global scaleFactor

    scaleType = args[0]
    scaleFactor = 1 + scaleFactor/100.0
    if scaleFactor == 0:
        scaleFactor = 1
    scaledImage = cv2.resize(img, None, fx=scaleFactor,
                             fy=scaleFactor, interpolation=cv2.INTER_LINEAR)
    cv2.imshow(windowName, scaledImage)


# Create Trackbar to choose percentage of scaling
cv2.createTrackbar(trackbarValue, windowName,
                   scaleFactor, maxScaleUp, scaleImage)

# Create Trackbar to choose type of scaling ( Up or down )
cv2.createTrackbar(trackbarType, windowName,
                   scaleType, maxType, scaleImage)

# Calling the function for the first time
# scaleImage(25)

# Start the loop to record
while True:
    key = cv2.waitKey(0)
    if key == 27:
        break

cv2.destroyAllWindows()
