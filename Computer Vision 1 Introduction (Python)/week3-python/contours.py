import cv2
import matplotlib.pyplot as plt
from dataPath import DATA_PATH
import numpy as np
import matplotlib
matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)
matplotlib.rcParams['image.cmap'] = 'gray'

imagePath = DATA_PATH + "images/Contour.png"
image = cv2.imread(imagePath)
imageCopy = image.copy()
# Convert to grayscale
imageGray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)


# Display image
plt.figure()
plt.subplot(121)
plt.imshow(image[:,:,::-1])
plt.title("Original Image");
plt.subplot(122)
plt.imshow(imageGray)
plt.title("Grayscale Image");
plt.show()

# Find all contours in the image
contours, hierarchy = cv2.findContours(imageGray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

print("Number of contours found = {}".format(len(contours)))
print("\nHierarchy : \n{}".format(hierarchy))

# Draw all the contours
cv2.drawContours(image, contours, -1, (0,255,0), 3);
cv2.imshow("image",image)
cv2.waitKey(0)

# Find external contours in the image
contours, hierarchy = cv2.findContours(imageGray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours found = {}".format(len(contours)))
image = imageCopy.copy()
# Draw all the contours
cv2.drawContours(image, contours, -1, (0,255,0), 3);
cv2.imshow("image",image)
cv2.waitKey(0)

# Draw only the 3rd contour
# Note that right now we do not know
# the numbering of contour in terms of the shapes
# present in the figure
image = imageCopy.copy()
cv2.drawContours(image, contours[2], -1, (0,0,255), 3);
cv2.imshow("image",image)
cv2.waitKey(0)

# Find all contours in the image
contours, hierarchy = cv2.findContours(imageGray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# Draw all the contours
cv2.drawContours(image, contours, -1, (0,255,0), 3);

for cnt in contours:
    # We will use the contour moments
    # to find the centroid
    M = cv2.moments(cnt)
    x = int(round(M["m10"]/M["m00"]))
    y = int(round(M["m01"]/M["m00"]))

    # Mark the center
    cv2.circle(image, (x,y), 10, (255,0,0), -1);

cv2.imshow("image",image)
cv2.waitKey(0)

for index,cnt in enumerate(contours):
    # We will use the contour moments
    # to find the centroid
    M = cv2.moments(cnt)
    x = int(round(M["m10"]/M["m00"]))
    y = int(round(M["m01"]/M["m00"]))
    
    # Mark the center
    cv2.circle(image, (x,y), 10, (255,0,0), -1);
    
    # Mark the contour number
    cv2.putText(image, "{}".format(index + 1), (x+40, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2);

imageCopy = image.copy()

cv2.imshow("image",image)
cv2.waitKey(0)

for index,cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    print("Contour #{} has area = {} and perimeter = {}".format(index+1,area,perimeter))

image = imageCopy.copy()
for cnt in contours:
    # Vertical rectangle
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,255), 2)

cv2.imshow("image",image)
cv2.waitKey(0)

image = imageCopy.copy()
for cnt in contours:
    # Rotated bounding box
    box = cv2.minAreaRect(cnt)
    boxPts = np.int0(cv2.boxPoints(box))
    # Use drawContours function to draw
    # rotated bounding box
    cv2.drawContours(image, [boxPts], -1, (0,255,255), 2)

cv2.imshow("image",image)
cv2.waitKey(0)

image = imageCopy.copy()
for cnt in contours:
    # Fit a circle
    ((x,y),radius) = cv2.minEnclosingCircle(cnt)
    cv2.circle(image, (int(x),int(y)), int(round(radius)), (125,125,125), 2)

cv2.imshow("image",image)
cv2.waitKey(0)

image = imageCopy.copy()
for cnt in contours:
    # Fit an ellipse
    # We can fit an ellipse only
    # when our contour has minimum
    # 5 points
    if len(cnt) < 5:
        continue
    ellipse = cv2.fitEllipse(cnt)
    cv2.ellipse(image, ellipse, (255,0,125), 2)

cv2.imshow("image",image)
cv2.waitKey(0)
