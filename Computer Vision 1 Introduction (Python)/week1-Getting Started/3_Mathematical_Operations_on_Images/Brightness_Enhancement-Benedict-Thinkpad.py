# Import libraries
import cv2
import numpy as np
from dataPath import DATA_PATH

# Read image
image = cv2.imread(DATA_PATH+"images/boy.jpg")

brightnessOffset = 50

# Add the offset for increasing brightness
brightHigh = image + brightnessOffset

# Display the outputs
#cv2.imshow("original Image",image)
#cv2.imshow("High Brightness",brightHigh)
#cv2.waitKey(0)
cv2.imwrite("../results/highBrightness.png",brightHigh)

print("Original Image Datatype : {}".format(image.dtype))
print("Brightness Image Datatype : {}\n".format(brightHigh.dtype))

print("Original Image Highest Pixel Intensity : {}".format(image.max()))
print("Brightness Image Highest Pixel Intensity : {}".format(brightHigh.max()))
