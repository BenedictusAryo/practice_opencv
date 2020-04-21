import cv2
from dataPath import DATA_PATH

filename = DATA_PATH + "images/capsicum.jpg"
bgr = cv2.imread(filename)

# convert from bgr to LAB format
labImage = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)

cv2.imshow("L Channel",labImage[:,:,0])
cv2.imshow("A Channel",labImage[:,:,1])
cv2.imshow("B Channel",labImage[:,:,2])
cv2.waitKey(0)
