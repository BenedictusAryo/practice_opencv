import cv2
import numpy as np
from dataPath import DATA_PATH

img = cv2.imread(DATA_PATH+"images/gaussian-noise.png")

# Check for invalid input
if img is None:
    print("Could not open or find the image")

# diameter of the pixel neighbourhood used during filtering
dia=15;

# Larger the value the distant colours will be mixed together
# to produce areas of semi equal colors
sigmaColor=80

# Larger the value more the influence of the farther placed pixels
# as long as their colors are close enough
sigmaSpace=80

#Apply bilateralFilter
result = cv2.bilateralFilter(img, dia, sigmaColor, sigmaSpace)

cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.imshow("Bilateral Blur Result", result)
cv2.waitKey(0)
