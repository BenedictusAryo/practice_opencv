# Import cv2 module
import cv2
from dataPath import DATA_PATH

# Path to the image we are going to read
# This can be an absolute or relative path
# Here we are using a relative path
imageName = DATA_PATH+"images/boy.jpg"

# Load the image
image = cv2.imread(imageName, cv2.IMREAD_COLOR)

# Draw a line
imageLine = image.copy()

# The line starts from (200,80) and ends at (280,80)
# The color of the line is RED (Recall that OpenCV uses BGR format)
# Thickness of line is 5px
# Linetype is cv2.LINE_AA

# We are using semicolon ';' to supress the output
# of the following statement

cv2.line(imageLine, (200, 80), (280, 80), (0, 255, 0), thickness=3, lineType=cv2.LINE_AA);

# Display the image
#cv2.imshow("Image Line",imageLine)
#cv2.waitKey(0)
cv2.imwrite("../results/imageLine.png",imageLine)
