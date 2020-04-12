import cv2
import numpy as np
from dataPath import DATA_PATH

filename = DATA_PATH+"images/truth.png"

image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

# Check for invalid input
if image is None:  
    print("Could not open or find the image")

# Apply sobel filter along x direction
sobelx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
# Apply sobel filter along y direction
sobely = cv2.Sobel(image,cv2.CV_32F,0,1)

# Normalize image for display
cv2.normalize(sobelx,
                dst = sobelx,
                alpha = 0,
                beta = 1,
                norm_type = cv2.NORM_MINMAX,
                dtype = cv2.CV_32F)
cv2.normalize(sobely,
                dst = sobely,
                alpha = 0,
                beta = 1,
                norm_type = cv2.NORM_MINMAX,
                dtype = cv2.CV_32F)
cv2.imshow("Sobel X Gradients", sobelx)
cv2.imshow("Sobel Y Gradients", sobely)
cv2.waitKey(0)


