import cv2
import numpy as np
from dataPath import DATA_PATH

filename = DATA_PATH+"images/gaussian-noise.png"

# Load an image
img = cv2.imread(filename)

# Apply box filter - kernel size 3
dst1=cv2.blur(img,(3,3),(-1,-1))

# Apply box filter - kernel size 7
dst2=cv2.blur(img,(7,7),(-1,-1))

cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.imshow("Box Blur Result 1 : KernelSize = 3", dst1)
cv2.waitKey(0)
cv2.imshow("Box Blur Result 1 : KernelSize = 7", dst2)
cv2.waitKey(0)


