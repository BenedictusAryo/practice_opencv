import cv2
import numpy as np
from dataPath import DATA_PATH

img = cv2.imread(DATA_PATH+"images/sample.jpg", cv2.IMREAD_GRAYSCALE)

kernelSize = 3

# Applying laplacian
img1 = cv2.GaussianBlur(img,(3,3),0,0)
laplacian = cv2.Laplacian(img1, cv2.CV_32F, ksize = kernelSize,
                            scale = 1, delta = 0)

# Normalize results
cv2.normalize(laplacian,
                dst = laplacian,
                alpha = 0,
                beta = 1,
                norm_type = cv2.NORM_MINMAX,
                dtype = cv2.CV_32F)

cv2.imshow("Laplacian", laplacian)
cv2.waitKey(0)
