# Import libraries
import cv2
import numpy as np
from dataPath import DATA_PATH

# Read image
image = cv2.imread(DATA_PATH+"images/boy.jpg")

# Let's see what image we are dealing with
#cv2.imshow("Image",image)
#cv2.waitKey(0)

# First let us create a copy of the original image 
copiedImage = image.copy()
#cv2.imshow("copiedImage",copiedImage)
#cv2.waitKey(0)

copyRoi = image[40:200,180:320]

# Find height and width of the ROI
roiHeight,roiWidth = copyRoi.shape[:2]

# Copy to left of Face
copiedImage[40:40+roiHeight, 10:10+roiWidth] = copyRoi
# Copy to right of Face
copiedImage[40:40+roiHeight, 330:330+roiWidth] = copyRoi

# Display the output
#cv2.imshow("Original Image",image)
#cv2.imshow("Output Image",copiedImage)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.imwrite("../results/outputImage.png",copiedImage)
