# Import cv2 module
import cv2
from dataPath import DATA_PATH

# Path to the image we are going to read
# This can be an absolute or relative path
# Here we are using a relative path
imageName = DATA_PATH+"images/boy.jpg"

# Load the image
image = cv2.imread(imageName, cv2.IMREAD_COLOR)

# Put text into image
imageText = image.copy()
text = "I am studying"
fontScale = 1.5
fontFace = cv2.FONT_HERSHEY_COMPLEX
fontColor = (250, 10, 10)
fontThickness = 2
cv2.putText(imageText, text, (20, 350), fontFace, fontScale, fontColor, fontThickness, cv2.LINE_AA);

# Display the image
#cv2.imshow("Image Text",imageText)
#cv2.waitKey(0)
cv2.imwrite("../results/imageText.png",imageText)

#Issues in Text annotation

#Solution 1: Get font size from pixel height of text

pixelHeight = 20

# Calculate the fontScale
fontScale = cv2.getFontScaleFromHeight(fontFace, pixelHeight, fontThickness)
print("fontScale = {}".format(fontScale))

imageTextFontScale = image.copy()
cv2.putText(imageTextFontScale, text, (20, 350), fontFace, fontScale, fontColor, fontThickness, cv2.LINE_AA);
# Display the image
#cv2.imshow("Image Text FontScale",imageTextFontScale)
#cv2.waitKey(0)
cv2.imwrite("../results/imageTextFontScale.png",imageTextFontScale)

#Solution 2: Get height and width of text

imageGetTextSize = image.copy()
imageHeight, imageWidth=imageGetTextSize.shape[:2]

# Get the text box height and width and also the baseLine
textSize, baseLine = cv2.getTextSize(text,fontFace,fontScale,fontThickness)
textWidth,textHeight = textSize
print("TextWidth = {}, TextHeight = {}, baseLine = {}".format(textWidth, textHeight, baseLine))

# Get the coordinates of text box bottom left corner
# The xccordinate will be such that the text is centered
xcoordinate = (imageWidth - textWidth)//2
# The y coordinate will be such that the entire box is just 10 pixels above the bottom of image
ycoordinate = (imageHeight - baseLine - 10)
print("TextBox Bottom Left = ({},{})".format(xcoordinate,ycoordinate))

# Draw the Canvas using a filled rectangle
canvasColor = (255, 255, 255)
canvasBottomLeft = (xcoordinate,ycoordinate+baseLine)
canvasTopRight = (xcoordinate+textWidth, ycoordinate-textHeight)
cv2.rectangle(imageGetTextSize, canvasBottomLeft, canvasTopRight, canvasColor, thickness=-1);
print("Canvas Bottom Left = {}, Top Right = {}".format(canvasBottomLeft,canvasTopRight))

# Now draw the baseline ( just for reference )
lineThickness = 2
lineLeft = (xcoordinate, ycoordinate)
lineRight = (xcoordinate+textWidth, ycoordinate)
lineColor = (0,255,0)
cv2.line(imageGetTextSize, lineLeft, lineRight, lineColor, thickness = lineThickness, lineType=cv2.LINE_AA);

# Finally Draw the text
cv2.putText(imageGetTextSize, text, (xcoordinate,ycoordinate), fontFace, fontScale, fontColor, fontThickness, cv2.LINE_AA);

# Display the Output Image
#cv2.imshow("Image Get TextSize",imageGetTextSize)
#cv2.waitKey(0)
cv2.imwrite("../results/imageGetTextSize.png",imageGetTextSize)
