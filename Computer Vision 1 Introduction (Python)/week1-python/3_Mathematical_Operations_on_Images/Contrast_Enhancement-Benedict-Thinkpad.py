# Import libraries
import cv2
import numpy as np
from dataPath import DATA_PATH

# Read image
image = cv2.imread(DATA_PATH+"images/boy.jpg")

contrastPercentage = 30

# Multiply with scaling factor to increase contrast
contrastHigh = image * (1+contrastPercentage/100)

# Display the outputs
#cv2.imshow("Original Image",image)
#cv2.imshow("High Contrast",contrastHigh)
#cv2.waitKey(0)
cv2.imwrite("../results/highContrast.png",contrastHigh)

print("Original Image Datatype : {}".format(image.dtype))
print("Contrast Image Datatype : {}".format(contrastHigh.dtype))

print("Original Image Highest Pixel Intensity : {}".format(image.max()))
print("Contrast Image Highest Pixel Intensity : {}".format(contrastHigh.max()))

contrastPercentage = 30

# Clip the values to [0,255] and change it back to uint8 for display
contrastImage = image * (1+contrastPercentage/100)
clippedContrastImage = np.clip(contrastImage, 0, 255)
contrastHighClippedUint8 = np.uint8(clippedContrastImage)

# Convert the range to [0,1] and keep it in float format
contrastHighNormalized = (image * (1+contrastPercentage/100))/255
contrastHighNormalized01Clipped = np.clip(contrastHighNormalized,0,1)

#cv2.imshow("Original Image",image)
#cv2.imshow("converted back to uint8",contrastHighClippedUint8)
#cv2.imshow("Normalized float to [0, 1]",contrastHighNormalized01Clipped)
#cv2.waitKey(0)
cv2.imwrite("../results/contrastHighClippedUint8.png",contrastHighClippedUint8)
cv2.imwrite("../results/contrastHighNormalized01Clipped.png",contrastHighNormalized01Clipped)
