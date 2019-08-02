# Import libraries
import cv2
import numpy as np
from dataPath import DATA_PATH

# Read image
image = cv2.imread(DATA_PATH+"images/boy.jpg")

# Let's see what image we are dealing with
#cv2.imshow("Image",image)
#cv2.waitKey(0)

# Create an empty image of same size as the original
mask1 = np.zeros_like(image)
#cv2.imshow("Mask",mask1)
#cv2.waitKey(0)
cv2.imwrite("../results/mask.png",mask1)

mask1[50:200,170:320] = 255
print(mask1.dtype)
#cv2.imshow("Original Image",image)
#cv2.imshow("Mask",mask1)
#cv2.waitKey(0)

mask2 = cv2.inRange(image, (0,0,150), (100,100,255))
#cv2.imshow("Original Image",image)
#cv2.imshow("Masked Image",mask2)
#cv2.waitKey(0)
cv2.imwrite("../results/maskedImage.png",mask2)
