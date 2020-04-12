import cv2
import numpy as np
from dataPath import DATA_PATH

filename = DATA_PATH+"images/sample.jpg"

image = cv2.imread(filename)

if image is None:
    print("Image not read")

# Sharpen kernel
sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int")

# Using 2D filter by applying the sharpening kernel
sharpenOutput = cv2.filter2D(image, -1, sharpen)

cv2.imshow("Original Image", image)
cv2.imshow("Sharpening Result", sharpenOutput)
cv2.waitKey(0)

