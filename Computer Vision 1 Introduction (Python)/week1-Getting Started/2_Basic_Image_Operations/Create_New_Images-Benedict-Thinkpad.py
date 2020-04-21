# Import libraries
import cv2
import numpy as np
from dataPath import DATA_PATH

# Read image
image = cv2.imread(DATA_PATH+"images/boy.jpg")

# Let's see what image we are dealing with
#cv2.imshow("Input image",image)

# Create a new image by copying the already present image using the copy operation
imageCopy = image.copy()

# Create an empty matrix
emptyMatrix = np.zeros((100,200,3),dtype='uint8')
#cv2.imshow("Empty matrix",emptyMatrix)
#cv2.waitKey(0)
cv2.imwrite("../results/emptyMatrix.png",emptyMatrix)

emptyMatrix = 255*np.ones((100,200,3),dtype='uint8')
#cv2.imshow("Empty matrix",emptyMatrix)
cv2.imwrite("../results/emptyMatrix1.png",emptyMatrix)

emptyOriginal = 100*np.ones_like(image)
#cv2.imshow("Empty Original",emptyOriginal)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.imwrite("../results/emptyOriginal.png",emptyOriginal)
