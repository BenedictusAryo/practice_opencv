# Import libraries
import cv2
import numpy as np
from dataPath import DATA_PATH

# Path of the image to be loaded
# Here we are supplying a relative path
imagePath = DATA_PATH + "/images/musk.jpg"

# Read the image
img = cv2.imread(imagePath)
print("image Dimension ={}".format(img.shape))
