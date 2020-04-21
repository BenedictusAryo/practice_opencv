import cv2, time
import numpy as np
from dataPath import DATA_PATH
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

# Read an image in grayscale
imagePath = DATA_PATH + '/images/threshold.png'
src = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

# Set threshold and maximum value
thresh = 100
maxValue = 255

def thresholdUsingLoop(src, thresh, maxValue):
    # Create a output image
    dst = src.copy()
    height,width = src.shape[:2]

    # Loop over rows
    for i in range(height):
        # Loop over columns
        for j in range(width):
            if src[i,j] > thresh:
                dst[i,j] = maxValue
            else:
                dst[i,j] = 0

    return dst

t = time.time()
dst = thresholdUsingLoop(src, thresh, maxValue)
print("Time taken = {} seconds".format(time.time() - t))

plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(src);plt.title("Original Image");
plt.subplot(122);plt.imshow(dst);plt.title("Thresholded Image");
plt.show()

def thresholdUsingVectors(src, thresh, maxValue):
    # Create a black output image ( all zeros )
    dst = np.zeros_like(src)

    # Find pixels which have values>threshold value
    thresholdedPixels = src>thresh

    # Assign those pixels maxValue
    dst[thresholdedPixels] = maxValue

    return dst

t = time.time()
dst = thresholdUsingVectors(src, thresh, maxValue)
print("Time taken = {} seconds".format(time.time() - t))

plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(src);plt.title("Original Image");
plt.subplot(122);plt.imshow(dst);plt.title("Thresholded Image");
plt.show()

t = time.time()
th, dst = cv2.threshold(src, thresh, maxValue, cv2.THRESH_BINARY)
print("Time taken = {} seconds".format(time.time() - t))

plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(src);plt.title("Original Image");
plt.subplot(122);plt.imshow(dst);plt.title("Thresholded Image");
plt.show()

time_opencv = 0
time_loops = 0
time_vector = 0
n_samples = 10
for i in range(n_samples):

    t = time.time()
    dst = thresholdUsingLoop(src, thresh, maxValue)
    time_loops += time.time() - t

    t = time.time()
    dst = thresholdUsingVectors(src, thresh, maxValue)
    time_vector += time.time() - t

    t = time.time()
    th, dst = cv2.threshold(src, thresh, maxValue, cv2.THRESH_BINARY)
    time_opencv += time.time() - t

print("Average time taken by For Loop Code = {} seconds".format(time_loops/n_samples))
print("Average time taken by Vectorized Code = {} seconds".format(time_vector/n_samples))
print("Average time taken by OpenCV Code = {} seconds".format(time_opencv/n_samples))

thresh = 100
maxValue = 150

th, dst_bin = cv2.threshold(src, thresh, maxValue, cv2.THRESH_BINARY)

th, dst_bin_inv = cv2.threshold(src, thresh, maxValue, cv2.THRESH_BINARY_INV)

th, dst_trunc = cv2.threshold(src, thresh, maxValue, cv2.THRESH_TRUNC)

th, dst_to_zero = cv2.threshold(src, thresh, maxValue, cv2.THRESH_TOZERO)

th, dst_to_zero_inv = cv2.threshold(src, thresh, maxValue, cv2.THRESH_TOZERO_INV)

print("Threshold Value = {}, Max Value = {}".format(thresh, maxValue))
plt.figure(figsize=[20,12])
plt.subplot(231);plt.imshow(src, cmap='gray', vmin=0, vmax=255);plt.title("Original Image");
plt.subplot(232);plt.imshow(dst_bin, cmap='gray', vmin=0, vmax=255);plt.title("Threshold Binary");
plt.subplot(233);plt.imshow(dst_bin_inv, cmap='gray', vmin=0, vmax=255);plt.title("Threshold Binary Inverse");
plt.subplot(234);plt.imshow(dst_trunc, cmap='gray', vmin=0, vmax=255);plt.title("Threshold Truncate");
plt.subplot(235);plt.imshow(dst_to_zero, cmap='gray', vmin=0, vmax=255);plt.title("Threshold To Zero");
plt.subplot(236);plt.imshow(dst_to_zero_inv, cmap='gray', vmin=0, vmax=255);plt.title("Threshold To Zero Inverse");
plt.show()
