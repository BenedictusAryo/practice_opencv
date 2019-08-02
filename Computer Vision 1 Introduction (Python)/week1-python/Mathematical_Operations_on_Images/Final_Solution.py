# Import libraries
import cv2
import numpy as np
from dataPath import DATA_PATH

# Read image
image = cv2.imread(DATA_PATH+"images/boy.jpg")

brightnessOffset = 50

# Add the offset for increasing brightness
brightHighOpenCV = cv2.add(image, np.ones(image.shape,dtype='uint8')*brightnessOffset)

brightHighInt32 = np.int32(image) + brightnessOffset
brightHighInt32Clipped = np.clip(brightHighInt32,0,255)

# Display the outputs
#cv2.imshow("original image",image)
#cv2.imshow("using cv2.add function",brightHighOpenCV)
#cv2.imshow("Using numpy and clipping", brightHighInt32Clipped)
#cv2.waitKey(0)
cv2.imwrite("../results/brightHighOpenCV.png",brightHighOpenCV)
cv2.imwrite("../results/brightHighInt32Clipped.png",brightHighInt32Clipped)

# Add the offset for increasing brightness
brightHighFloat32 = np.float32(image) + brightnessOffset
brightHighFloat32NormalizedClipped = np.clip(brightHighFloat32/255,0,1)

brightHighFloat32ClippedUint8 = np.uint8(brightHighFloat32NormalizedClipped*255)

# Display the outputs
#cv2.imshow("original image",image)
#cv2.imshow("Using np.float32 and clipping",brightHighFloat32NormalizedClipped)
#cv2.imshow("Using int->float->int and clipping", brightHighFloat32ClippedUint8)
#cv2.waitKey(0)
cv2.imwrite("../results/brightHighFloat32NormalizedClipped.png",brightHighFloat32NormalizedClipped)
cv2.imwrite("../results/brightHighFloat32ClippedUint8.png",brightHighFloat32ClippedUint8)
