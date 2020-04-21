# Import libraries
import cv2
import numpy as np
from dataPath import DATA_PATH

imagePath = DATA_PATH + "/images/number_zero.jpg"

# Read image in Grayscale format
testImage = cv2.imread(imagePath,0)

#Access a region
test_roi = testImage[0:2,0:4]
print("Original Matrix\n{}\n".format(testImage))
print("Selected Region\n{}\n".format(test_roi))

testImage[0:2,0:4] = 111
print("Modified Matrix\n{}\n".format(testImage))

