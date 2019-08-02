# Import libraries
import cv2
import numpy as np
from dataPath import DATA_PATH

# Load the Face Image
faceImagePath = DATA_PATH + "/images/musk.jpg"
faceImage = cv2.imread(faceImagePath)

#cv2.imshow("Face",faceImage)
cv2.imwrite("../results/face.png",faceImage)

# Load the Sunglass image with Alpha channel
# (http://pluspng.com/sunglass-png-1104.html)
glassimagePath = DATA_PATH + "/images/sunglass.png"
glassPNG = cv2.imread(glassimagePath,-1)

# Resize the image to fit over the eye region
glassPNG = cv2.resize(glassPNG,(300,100))
print("image Dimension ={}".format(glassPNG.shape))

# Separate the Color and alpha channels
glassBGR = glassPNG[:,:,0:3]
glassMask1 = glassPNG[:,:,3]

# Display the images for clarity
#cv2.imshow("Sunglasses Color Channels",glassBGR)
#cv2.imshow("Sunglass Alpha Channel",glassMask1)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.imwrite("../results/glassBGR.png",glassBGR)
cv2.imwrite("../results/glassMask1.png",glassMask1)
