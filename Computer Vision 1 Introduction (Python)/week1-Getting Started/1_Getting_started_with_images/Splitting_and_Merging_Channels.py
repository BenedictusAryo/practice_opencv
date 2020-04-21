# Import libraries
import cv2
import numpy as np
from dataPath import DATA_PATH

# Path of the image to be loaded
# Here we are supplying a relative path
imagePath = DATA_PATH + "/images/musk.jpg"

# Read the image
img = cv2.imread(imagePath)

# Split the image into the B,G,R components
b,g,r = cv2.split(img)

# Show the channels
#cv2.imshow("Blue Channel",b)
#cv2.imshow("Green Channel",g)
#cv2.imshow("Red Channel",r)
cv2.imwrite("../results/blueChannel.png",b)
cv2.imwrite("../results/greenChannel.png",g)
cv2.imwrite("../results/redChannel.png",r)

# Merge the individual channels into a BGR image
mergedOutput = cv2.merge((b,g,r))
# Show the merged output
#cv2.imshow("Merged Output",mergedOutput)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.imwrite("../results/mergedOutput.png",mergedOutput)
