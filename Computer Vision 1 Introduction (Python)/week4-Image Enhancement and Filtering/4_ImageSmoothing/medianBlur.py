import cv2
import numpy as np
from dataPath import DATA_PATH

filename = DATA_PATH + "images/salt-and-pepper.png"

img = cv2.imread(filename)

# Defining the kernel size
kernelSize = 5

# Performing Median Blurring and store it in numpy array "medianBlurred"
medianBlurred = cv2.medianBlur(img,kernelSize)

cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.imshow("Median Blur Result : KernelSize = 5", medianBlurred)
cv2.waitKey(0)

