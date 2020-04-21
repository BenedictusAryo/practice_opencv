# Import libraries
import cv2
import numpy as np
from dataPath import DATA_PATH

# Read image
image = cv2.imread(DATA_PATH+"images/boy.jpg")

# Let's see what image we are dealing with
#cv2.imshow("Image",image)
#cv2.waitKey(0)

# Crop out a rectangle
# x coordinates = 170 to 320
# y coordinates = 40 to 200
crop = image[40:200,170:320]
#cv2.imshow("Cropped Image",crop)
#cv2.waitKey(0)
cv2.imwrite("../results/croppedImage.png",crop)
