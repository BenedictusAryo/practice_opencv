# How to use the Mouse in OpenCV

# We can detect mouse events like left-click, right-click or position of the mouse on the window using OpenCV.
# For doing that, we need to create a named window and assign a callback function to the window.
# We will see how it is done in the code.
# The code given below draws a circle on the image.
# You first mark the center of the circle and then drag the mouse according to the radius desired.
# Multiple circles can be drawn.
# 'c' is used to clear the screen (the circles) and pressing 'ESC' terminates the program.
# We will see the detailed code in the code video. For now, let's just focus on the callback function.

import cv2
import math
# Lists to store the points
center = []
circumference = []


def drawCircle(action, x, y, flags, userdata):
    # Referencing global variables
    global center, circumference
    # Action to be taken when left mouse button is pressed
    if action == cv2.EVENT_LBUTTONDOWN:
        center = [(x, y)]
        # Mark the center
        cv2.circle(source, center[0], 1,
                   (255, 255, 0), 2, cv2.LINE_AA)

        # Action to be taken when left mouse button is released
    elif action == cv2.EVENT_LBUTTONUP:
        circumference = [(x, y)]
        # Calculate radius of the circle
        radius = math.sqrt(math.pow(center[0][0] -
                                    circumference[0][0], 2) +
                           math.pow(center[0][1] - circumference[0][1], 2))
        # Draw the circle
        cv2.circle(source, center[0], int(radius), (0, 255, 0),
                   2, cv2.LINE_AA)
        cv2.imshow("Window", source)

# drawCircle the callback function is called when there is a mouse event like left click ( indicated by EVENT_LBUTTONDOWN ).
# The coordinates relative to the namedWindow is captured by this function in the variables (x,y).
# The function records the points of the circle’s center and a point on the circumference, hence allowing us to draw the desired circle on the image.


# This is the main function. We read the image, setup the mouse callback function and loops till the ESC character is pressed.
# We use the setMouseCallback function to assign the function drawCircle defined above to the window named "Window"

source = cv2.imread("sample.jpg", 1)
# Make a dummy image, will be useful to clear the drawing
dummy = source.copy()
cv2.namedWindow("Window")
# highgui function called when mouse events occur
cv2.setMouseCallback("Window", drawCircle)
k = 0
# loop until escape character is pressed
while k != 27:
    cv2.imshow("Window", source)
    cv2.putText(source, '''Choose center, and drag, 
    Press ESC to exit and c to clear''',
                (10, 30), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                0.7, (255, 255, 255), 2)
    k = cv2.waitKey(0) & 0xFF
    # Another way of cloning
    if k == 99:
        source = dummy.copy()

cv2.destroyAllWindows()
