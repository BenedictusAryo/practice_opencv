# Import libraries
import cv2
import numpy as np
from dataPath import DATA_PATH

imagePath = DATA_PATH + "/images/number_zero.jpg"

# Read image in Grayscale format
testImage = cv2.imread(imagePath,0)

#access the first element
print(testImage[0,0])

#Modifying pixel value
testImage[0,0]=200
print(testImage)
