import cv2
import matplotlib.pyplot as plt
import numpy as np
from dataPath import DATA_PATH

filename = DATA_PATH + "images/girl.jpg"

original = cv2.imread(filename)
img = np.copy(original)

# pivot points for X-Coordinates
originalValue = np.array([0, 50, 100, 150, 200, 255])

# Changed points on Y-axis for each channel
bCurve = np.array([0, 80, 150, 190, 220, 255])
rCurve = np.array([0, 20,  40,  75, 150, 255])

# Create a LookUp Table
fullRange = np.arange(0,256)
rLUT = np.interp(fullRange, originalValue, rCurve )
bLUT = np.interp(fullRange, originalValue, bCurve )

# Get the blue channel and apply the mapping
bChannel = img[:,:,0]
bChannel = cv2.LUT(bChannel, bLUT)
img[:,:,0] = bChannel

# Get the red channel and apply the mapping
rChannel = img[:,:,2]
rChannel = cv2.LUT(rChannel, rLUT)
img[:,:,2] = rChannel

# show and save the ouput
combined = np.hstack([original,img])

cv2.imshow("Cooling filter output", combined)
cv2.waitKey(0)
