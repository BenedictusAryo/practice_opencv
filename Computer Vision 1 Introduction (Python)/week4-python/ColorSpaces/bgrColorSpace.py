# Import module
import cv2
import numpy as np
from dataPath import DATA_PATH

#read the image in BGR format
bgr = cv2.imread(DATA_PATH+"images/capsicum.jpg")

cv2.imshow("bgr",bgr)
cv2.waitKey(0)

cv2.imshow("Blue Channel",bgr[:,:,0])
cv2.imshow("Green Channel",bgr[:,:,1])
cv2.imshow("Red Channel",bgr[:,:,2])
cv2.waitKey(0)


