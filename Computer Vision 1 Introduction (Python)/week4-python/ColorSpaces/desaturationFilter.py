import cv2
import numpy as np
from dataPath import DATA_PATH

filename = DATA_PATH+"images/girl.jpg"
img = cv2.imread(filename)

# Specify scaling factor
saturationScale = 0.01

# Convert to HSV color space
hsvImage = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

# Convert to float32
hsvImage = np.float32(hsvImage)

# Split the channels
H, S, V = cv2.split(hsvImage)

# Multiply S channel by scaling factor and clip the values to stay in 0 to 255
S = np.clip(S * saturationScale , 0, 255)

# Merge the channels and show the output
hsvImage = np.uint8( cv2.merge([H, S, V]) )
imSat = cv2.cvtColor(hsvImage, cv2.COLOR_HSV2BGR)

cv2.imshow("Original Image",img)
cv2.waitKey(0)

cv2.imshow("Desaturated Image", imSat)
cv2.waitKey(0)

cv2.destroyAllWindows()
