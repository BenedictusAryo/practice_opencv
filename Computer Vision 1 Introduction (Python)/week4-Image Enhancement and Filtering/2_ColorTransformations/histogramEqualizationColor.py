import cv2
from dataPath import DATA_PATH
import numpy as np

# Read color image
filename = DATA_PATH+"images/dark-flowers.jpg"
im = cv2.imread(filename)
imEq = np.zeros_like(im)

# Peform histogram equalization on each channel separately
for i in range(0,3):
    imEq[:,:,i] = cv2.equalizeHist(im[:,:,i])

cv2.imshow("Original Image",im)
cv2.waitKey(0)
cv2.imshow("Histogram Equalized",imEq)
cv2.waitKey(0)

# Read color image
filename = DATA_PATH+"images/dark-flowers.jpg"
im = cv2.imread(filename)

# Convert to HSV
imhsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

# Perform histogram equalization only on the V channel
imhsv[:,:,2] = cv2.equalizeHist(imhsv[:,:,2])

# Convert back to BGR format
imEq = cv2.cvtColor(imhsv, cv2.COLOR_HSV2BGR)

cv2.imshow("Original Image",im)
cv2.waitKey(0)
cv2.imshow("Histogram Equalized",imEq)
cv2.waitKey(0)
