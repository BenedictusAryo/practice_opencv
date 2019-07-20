# Import libraries
import cv2
import numpy as np
from dataPath import DATA_PATH
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

imagePath = DATA_PATH + "/images/number_zero.jpg"
# Read image in Grayscale format
testImage = cv2.imread(imagePath,0)

#Write the Image to Disk
cv2.imwrite("../results/test.jpg",testImage)

