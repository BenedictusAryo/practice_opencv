import cv2
from dataPath import DATA_PATH
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (50.0, 50.0)
matplotlib.rcParams['image.cmap'] = 'gray'
matplotlib.rcParams['axes.titlesize'] = 40
matplotlib.rcParams['image.interpolation'] = 'bilinear'

# Read image in grayscale format
filename = DATA_PATH+"images/dark-flowers.jpg"
im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

# Equalize histogram
imEq = cv2.equalizeHist(im)

cv2.imshow("Original Image", im)
cv2.waitKey(0)

cv2.imshow("Histogram Equalized", imEq)
cv2.waitKey(0)


plt.figure(figsize=(30,10))
plt.subplot(1,2,1)
plt.hist(im.ravel(),256,[0,256]);

plt.subplot(1,2,2)
plt.hist(imEq.ravel(),256,[0,256]);
plt.show()

