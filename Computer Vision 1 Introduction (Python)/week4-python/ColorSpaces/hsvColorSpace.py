# Import module
import cv2
import numpy as np
from dataPath import DATA_PATH
import matplotlib.pyplot as plt

#read the image in BGR format
bgr = cv2.imread(DATA_PATH+"images/capsicum.jpg")

# convert from bgr to HSV format
hsvImage = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

cv2.imshow("hsv",hsvImage)

cv2.imshow("Hue",hsvImage[:,:,0])
cv2.imshow("Saturation",hsvImage[:,:,1])
cv2.imshow("Value",hsvImage[:,:,2])
cv2.waitKey(0)

for i in range(0,7):
    # Create 50x50 HSV image with all zeros
    imhsv = np.zeros((50, 50, 3), dtype=np.uint8)

    # Set Hue = 0, Saturation = 0, Value = i x 40
    v = i * 40
    imhsv[:,:,:] = (0, 0, v)

    # Convert HSV to RGB
    imrgb = cv2.cvtColor(imhsv, cv2.COLOR_HSV2RGB)

    # Display image
    ax = plt.subplot(1, 7, i+1)
    plt.imshow(imrgb)
    plt.axis('off')
    ax.set_title('V='+ str(v), fontdict={'fontsize': 10, 'fontweight': 'medium'})

plt.show()

# Saturation test.
# Set brightness to 128, hue to 0, and change saturation

for i in range(0,7):
    # Create 50x50 HSV image with all zeros
    imhsv = np.zeros((50, 50, 3), dtype=np.uint8)

    # Set Hue = 0, Saturation = i * 40, Value = 128
    s = i * 40
    imhsv[:,:,:] = (0, s, 128)

    # Convert HSV to RGB
    imrgb = cv2.cvtColor(imhsv, cv2.COLOR_HSV2RGB)

    # Display image
    ax = plt.subplot(1, 7, i+1)
    plt.imshow(imrgb)
    plt.axis('off')
    ax.set_title('S='+ str(s), fontdict={'fontsize': 10, 'fontweight': 'medium'})


plt.show()

# Hue Test

for i in range(0,7):
    imhsv = np.zeros((50, 50, 3), dtype=np.uint8)

    # Set Hue = i x 30, Saturation = 128, and Value = 128.
    h = i * 30
    imhsv[:,:,:] = ( h, 128, 128)

    # Convert HSV to RGB
    imrgb = cv2.cvtColor(imhsv, cv2.COLOR_HSV2RGB)

    # Display image
    ax = plt.subplot(1, 7, i+1)
    plt.imshow(imrgb)
    plt.axis('off')
    ax.set_title('H='+ str(h), fontdict={'fontsize': 10, 'fontweight': 'medium'})

plt.show()
