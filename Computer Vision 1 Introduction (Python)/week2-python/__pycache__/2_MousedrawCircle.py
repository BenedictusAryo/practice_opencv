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
