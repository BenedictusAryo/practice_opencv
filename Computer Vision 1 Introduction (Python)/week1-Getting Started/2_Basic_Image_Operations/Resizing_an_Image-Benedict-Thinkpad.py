# Import libraries
import cv2
import numpy as np
from dataPath import DATA_PATH

# Read image
image = cv2.imread(DATA_PATH+"images/boy.jpg")

# Let's see what image we are dealing with
#cv2.imshow("Image",image)
#cv2.waitKey(0)

#Method 1 - Specify width and height

# Set rows and columns
resizeDownWidth = 300
resizeDownHeight = 200
resizedDown = cv2.resize(image, (resizeDownWidth, resizeDownHeight), interpolation= cv2.INTER_LINEAR)

# Mess up with the aspect ratio
resizeUpWidth = 600
resizeUpHeight = 900
resizedUp = cv2.resize(image, (resizeUpWidth, resizeUpHeight), interpolation= cv2.INTER_LINEAR)

#cv2.imshow("Original Image",image)
#cv2.imshow("Scaled Up Image",resizedUp)
#cv2.imshow("Scaled Down Image",resizedDown)
#cv2.waitKey(0)
cv2.imwrite("../results/resizedUp.png",resizedUp)
cv2.imwrite("../results/resizedDown.png",resizedDown)

#Method 2 - Specify Scaling Factor

# Scaling Down the image 1.5 times by specifying both scaling factors
scaleUpX = 1.5
scaleUpY = 1.5

# Scaling Down the image 0.6 times specifying a single scale factor.
scaleDown = 0.6

scaledDown = cv2.resize(image, None, fx= scaleDown, fy= scaleDown, interpolation= cv2.INTER_LINEAR)

scaledUp = cv2.resize(image, None, fx= scaleUpX, fy= scaleUpY, interpolation= cv2.INTER_LINEAR)

# We can also use the following syntax for displaying image
#cv2.imshow("Scaled Down Image",scaledDown)
#cv2.imshow("Scaled Up Image",scaledUp)
#cv2.waitKey(0)
cv2.imwrite("../results/scaledUp.png",scaledUp)
cv2.imwrite("../results/scaledDown.png",scaledDown)
