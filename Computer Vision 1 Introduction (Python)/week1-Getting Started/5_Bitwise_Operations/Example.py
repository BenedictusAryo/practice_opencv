# Import libraries
import cv2
import numpy as np
from dataPath import DATA_PATH

# Load the Face Image
faceImagePath = DATA_PATH + "/images/musk.jpg"
faceImage = cv2.imread(faceImagePath)
#cv2.imshow("Face Image",faceImage)
#cv2.waitKey(0)
# Make a copy
faceWithGlassesBitwise = faceImage.copy()

# Load the Sunglass image with Alpha channel
glassimagePath = DATA_PATH + "/images/sunglass.png"
glassPNG = cv2.imread(glassimagePath,-1)

# Resize the image to fit over the eye region
glassPNG = cv2.resize(glassPNG,(300,100))
print("image Dimension ={}".format(glassPNG.shape))

# Separate the Color and alpha channels
glassBGR = glassPNG[:,:,0:3]
glassMask1 = glassPNG[:,:,3]

# Display the images for clarity
#cv2.imshow("Color Channels",glassBGR)
#cv2.imshow("Alpha Channels",glassMask1)
#cv2.waitKey(0)
cv2.imwrite("../results/glassBGR.png",glassBGR)
cv2.imwrite("../results/glassMask1.png",glassMask1)

# Get the eye region from the face image
eyeROI= faceWithGlassesBitwise[150:250,140:440]

# Make the dimensions of the mask same as the input image.
# Since Face Image is a 3-channel image, we create a 3 channel image for the mask
glassMask = cv2.merge((glassMask1,glassMask1,glassMask1))

# Use the mask to create the masked eye region
eye = cv2.bitwise_and(eyeROI,cv2.bitwise_not(glassMask))

# Use the mask to create the masked sunglass region
sunglass = cv2.bitwise_and(glassBGR,glassMask)

# Combine the Sunglass in the Eye Region to get the augmented image
eyeRoiFinal = cv2.bitwise_or(eye, sunglass)

# Display the intermediate results
#cv2.imshow("Masked Eye Region",eye)
#cv2.imshow("Masked Sunglass",sunglass)
#cv2.imshow("Combined Eye Region",eyeRoiFinal)
#cv2.waitKey(0)
cv2.imwrite("../results/maskedEyeRegionBitwise.png",eye)
cv2.imwrite("../results/maskedSunglassBitwise.png",sunglass)
cv2.imwrite("../results/combinedEyeRegionBitwise.png",np.uint8(eyeRoiFinal))

# Replace the eye ROI with the output from the previous section
faceWithGlassesBitwise[150:250,140:440]=eyeRoiFinal

# Display the final result
#cv2.imshow("Original Image",faceImage)
#cv2.imshow("With Sunglasses",faceWithGlassesBitwise)
#cv2.waitKey(0)
cv2.imwrite("../results/faceWithGlassesBitwise.png",faceWithGlassesBitwise)
