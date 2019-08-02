# Import cv2 module
import cv2
from dataPath import DATA_PATH

# Path to the image we are going to read
# This can be an absolute or relative path
# Here we are using a relative path
imageName = DATA_PATH+"images/boy.jpg"

# Load the image
image = cv2.imread(imageName, cv2.IMREAD_COLOR)

# Draw an ellipse
# Note: Ellipse Centers and Axis lengths must be integers
imageEllipse = image.copy()
cv2.ellipse(imageEllipse, (250, 125), (100, 50), 0, 0, 360, (255, 0, 0), thickness=3, lineType=cv2.LINE_AA);
cv2.ellipse(imageEllipse, (250, 125), (100, 50), 90, 0, 360, (0, 0, 255), thickness=3, lineType=cv2.LINE_AA);

# Display the image
#cv2.imshow("Image Ellipse",imageEllipse)
#cv2.waitKey(0)
cv2.imwrite("../results/imageEllipse.png",imageEllipse)

# Draw an ellipse
# Note: Ellipse Centers and Axis lengths must be integers
imageEllipse = image.copy()
# Incomplete/Open ellipse
cv2.ellipse(imageEllipse, (250, 125), (100, 50), 0, 180, 360, (255, 0, 0), thickness=3, lineType=cv2.LINE_AA);
# Filled ellipse
cv2.ellipse(imageEllipse, (250, 125), (100, 50), 0, 0, 180, (0, 0, 255), thickness=-2, lineType=cv2.LINE_AA);

# Display the image
#cv2.imshow("Image Filled Ellipse",imageEllipse)
#cv2.waitKey(0)
cv2.imwrite("../results/imageFilledEllipse.png",imageEllipse)
