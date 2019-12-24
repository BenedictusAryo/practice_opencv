import cv2
import numpy as np
from dataPath import DATA_PATH

filename = DATA_PATH+"images/gaussian-noise.png"

img = cv2.imread(filename)

# Apply gaussian blur
dst1=cv2.GaussianBlur(img,(5,5),0,0)
dst2=cv2.GaussianBlur(img,(25,25),50,50)

lineType=4
fontScale=1

# Display images
cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.imshow("Gaussian Blur Result 1 : KernelSize = 5", dst1)
cv2.waitKey(0)
cv2.imshow("Gaussian Blur Result 2 : KernelSize = 25", dst2)
cv2.waitKey(0)


