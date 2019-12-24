import cv2
import numpy as np
from dataPath import DATA_PATH

filename = DATA_PATH + "images/capsicum.jpg"

bgr = cv2.imread(filename)

ycbImage = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)

cv2.imshow("Y Channel", ycbImage[:,:,0])
cv2.imshow("Cr Channel", ycbImage[:,:,1])
cv2.imshow("Cb Channel", ycbImage[:,:,2])
cv2.waitKey(0)
