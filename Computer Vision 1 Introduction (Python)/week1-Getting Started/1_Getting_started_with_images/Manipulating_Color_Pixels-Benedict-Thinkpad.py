# Import libraries
import cv2
import numpy as np
from dataPath import DATA_PATH

# Path of the image to be loaded
imagePath = DATA_PATH + "/images/number_zero.jpg"
testImage = cv2.imread(imagePath,1)

#Access color pixel
print(testImage[0,0])

#Modify Pixel
testImage[0,0] = (0,255,255)
#cv2.imshow("Zero 1",testImage)
cv2.imwrite("../results/zero1.png",testImage)

testImage[1,1] = (255,255,0)
#cv2.imshow("Zero 2",testImage)
cv2.imwrite("../results/zero2.png",testImage)

testImage[2,2] = (255,0,255)
#cv2.imshow("Zero 3",testImage)
cv2.imwrite("../results/zero3.png",testImage)

#Modify Region of Interest
testImage[0:3,0:3] = (255,0,0)
testImage[3:6,0:3] = (0,255,0)
testImage[6:9,0:3] = (0,0,255)

#cv2.imshow("Zero", testImage)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.imwrite("../results/zero.png",testImage)
