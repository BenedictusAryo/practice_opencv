# Import libraries
import cv2
import numpy as np
from dataPath import DATA_PATH

imagePath = DATA_PATH + "/images/number_zero.jpg"

# Read image in Grayscale format
testImage = cv2.imread(imagePath,0)
print(testImage)

#Image properties
print("Data type = {}\n".format(testImage.dtype))
print("Object type = {}\n".format(type(testImage)))
print("Image Dimensions = {}\n".format(testImage.shape))
