import cv2
import numpy as np
from dataPath import DATA_PATH
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

imageName = DATA_PATH + "images/dilation_example.jpg"

# Read the input image
image = cv2.imread(imageName)

# Check for an invalid input
if image is None:
    print("Could not open or find the image")
cv2.imshow("image",image)
cv2.waitKey(0)

# Get structuring element/kernel which will be used for dilation
kSize = (7,7)
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kSize)
cv2.imshow("image",kernel1*255)
cv2.waitKey(0)

# Apply dilate function on the input image
imageDilated = cv2.dilate(image, kernel1)

plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(image);plt.title("Original Image")
plt.subplot(122);plt.imshow(imageDilated);plt.title("Dilated Image");
plt.show()

# Get structuring element/kernel which will be used for dilation
kSize = (3,3)
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kSize)
cv2.imshow("image",255*kernel2)
cv2.waitKey(0)

# Apply dilate function on the input image
imageDilated1 = cv2.dilate(image, kernel2, iterations=1)
imageDilated2 = cv2.dilate(image, kernel2, iterations=2)

plt.figure(figsize=[20,20])
plt.subplot(131);plt.imshow(image);plt.title("Original Image")
plt.subplot(132);plt.imshow(imageDilated1);plt.title("Dilated Image Iteration 1");
plt.subplot(133);plt.imshow(imageDilated2);plt.title("Dilated Image Iteration 2");
plt.show()

# Image taken as input
imageName = DATA_PATH + "images/erosion_example.jpg"
image = cv2.imread(imageName, cv2.IMREAD_COLOR)
# Check for invalid input
if image is None:
    print("Could not open or find the image")
cv2.imshow("image",image)
cv2.waitKey(0)

# Eroding the image , decreases brightness of image
imageEroded = cv2.erode(image, kernel1)

plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(image);plt.title("Original Image")
plt.subplot(122);plt.imshow(imageEroded);plt.title("Eroded Image");
plt.show()


