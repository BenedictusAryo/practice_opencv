# Import cv2 module
import cv2
from dataPath import DATA_PATH

# Path to the image we are going to read
# This can be an absolute or relative path
# Here we are using a relative path
imageName = DATA_PATH+"images/boy.jpg"

# Load the image
image = cv2.imread(imageName, cv2.IMREAD_COLOR)

# Draw a rectangle (thickness is a positive integer)
imageRectangle = image.copy()
cv2.rectangle(imageRectangle, (170, 50), (300, 200), (255, 0, 255), thickness=5, lineType=cv2.LINE_8);

# Display the image
#cv2.imshow("Image Rectangle",imageRectangle)
#cv2.waitKey(0)
cv2.imwrite("../results/imageRectangle.png",imageRectangle)
