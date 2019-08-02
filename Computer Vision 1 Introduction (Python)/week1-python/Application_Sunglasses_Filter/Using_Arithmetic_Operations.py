# Import libraries
import cv2
import numpy as np
from dataPath import DATA_PATH

# Load the Face Image
faceImagePath = DATA_PATH + "/images/musk.jpg"
faceImage = cv2.imread(faceImagePath)

#cv2.imshow("Face",faceImage)

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

# Make the dimensions of the mask same as the input image.
# Since Face Image is a 3-channel image, we create a 3 channel image for the mask
glassMask = cv2.merge((glassMask1,glassMask1,glassMask1))

# Make the values [0,1] since we are using arithmetic operations
glassMask = np.uint8(glassMask/255)

# Make a copy
faceWithGlassesArithmetic = faceImage.copy()

# Get the eye region from the face image
eyeROI= faceWithGlassesArithmetic[150:250,140:440]

# Use the mask to create the masked eye region
maskedEye = cv2.multiply(eyeROI,(1-  glassMask ))

# Use the mask to create the masked sunglass region
maskedGlass = cv2.multiply(glassBGR,glassMask)

# Combine the Sunglass in the Eye Region to get the augmented image
eyeRoiFinal = cv2.add(maskedEye, maskedGlass)

#cv2.imshow("Masked Eye Region",maskedEye)
#cv2.imshow("Masked Sunglass Region",maskedGlass)
#cv2.imshow("Augmented Eye and Sunglass",eyeRoiFinal)
cv2.imwrite("../results/maskedEyeRegion.png",maskedEye)
cv2.imwrite("../results/maskedSunglassRegion.png",maskedGlass)
cv2.imwrite("../results/augmentedEyeAndSunglass.png",eyeRoiFinal)

# Replace the eye ROI with the output from the previous section
faceWithGlassesArithmetic[150:250,140:440]=eyeRoiFinal

#cv2.imshow("With Sunglasses",faceWithGlassesArithmetic)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.imwrite("../results/withSunglasses.png",faceWithGlassesArithmetic)
