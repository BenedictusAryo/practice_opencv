import cv2
import numpy as np
from dataPath import DATA_PATH

# Read color image
filename = DATA_PATH+"images/night-sky.jpg"
im = cv2.imread(filename)

# Convert to HSV
imhsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
imhsvCLAHE = imhsv.copy()

# Perform histogram equalization only on the V channel
imhsv[:,:,2] = cv2.equalizeHist(imhsv[:,:,2])

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
imhsvCLAHE[:,:,2] = clahe.apply(imhsvCLAHE[:,:,2])

# Convert back to BGR format
imEq = cv2.cvtColor(imhsv, cv2.COLOR_HSV2BGR)
imEqCLAHE = cv2.cvtColor(imhsvCLAHE, cv2.COLOR_HSV2BGR)

cv2.imshow("Original Image", im)
cv2.imshow("Histogram Equalized",imEq)
cv2.imshow("CLAHE", imEqCLAHE)
cv2.waitKey(0)
