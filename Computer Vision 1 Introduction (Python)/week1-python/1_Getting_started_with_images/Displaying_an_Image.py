# Import libraries
import cv2
import numpy as np
from dataPath import DATA_PATH

imagePath = DATA_PATH + "/images/number_zero.jpg"

# Read image in Grayscale format
testImage = cv2.imread(imagePath,0)

#Matplotlib's imshow
#cv2.imshow("Image",testImage)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.imwrite("../results/testImage.png",testImage)
