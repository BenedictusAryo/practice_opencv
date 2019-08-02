# Import cv2 module
import cv2
from dataPath import DATA_PATH

# Path to the image we are going to read
# This can be an absolute or relative path
# Here we are using a relative path
imageName = DATA_PATH+"images/boy.jpg"

# Load the image
image = cv2.imread(imageName, cv2.IMREAD_COLOR)

# Draw a circle
imageCircle = image.copy()
cv2.circle(imageCircle, (250, 125), 100, (0, 0, 255), thickness=5, lineType=cv2.LINE_AA);

# Display the image
#cv2.imshow("Image Circle",imageCircle)
#cv2.waitKey(0)
cv2.imwrite("../results/imageCircle.png",imageCircle)

# Draw a filled circle
imageFilledCircle = image.copy()
cv2.circle(imageFilledCircle, (250, 125), 100, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA);

# Display the image
#cv2.imshow("Image Filled Circle",imageFilledCircle)
#cv2.waitKey(0)
cv2.imwrite("../results/imageFilledCircle.png",imageFilledCircle)
