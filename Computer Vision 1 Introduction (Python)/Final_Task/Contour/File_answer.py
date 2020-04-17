'''
QUESTION
While designing high speed cars, it is very important to study the flow of air as it passes the vehicle. All the major vehicle manufacturers use Wind Tunnels to perform these experiments. 

You are a design engineer who has been provided some images of the experiments and based on those images, you have to carry out the CFD (Computational Fluid Dynamics) experiments to come up with an optimum design. 

While performing the analysis, you took a cross section across the center of the vehicle (refer to the image above). To perform the fluid and aerodynamic analysis, the entire space is to be divided into cells (meshing). It is also important to understand that computation power involved in performing the modelling and analysis is a major parameter to consider. So, you need a fine mesh (large number of small cells) to get accurate results but having a large number of cells will make the process computationally extensive.

Your fellow engineer tells you that you can ignore the interior of the vehicle and have a fine grid near the body of the vehicle since the contours in that region play major role in the aerodynamic analysis.

In the image provided above, blue region has a very fine mesh, grey region has a coarser mesh and the black region has the smallest number of cells (and they are the largest ones). The interior of the vehicle should NOT be meshed.

Taking into account the sample image provided above, how will you go about removing the interior of the vehicle and identify the region which requires fine meshing?

Hint: Use contours to identify the interior of the vehicle.
'''

'''
ANSWER: 
The region which requires fine meshing is basically the vehicle body and the wheels. Examining the image closely, it is a single contour; it is the outline of the car.

The following steps should be followed in sequence:
1. Read and load the image
2. Convert it to grayscale
3. Perform thresholding. Inverse binary thresholding was chosen because it gives the best result
4. Crop the thresholded image. The area of interest is between row 200 and row 500.
5. Perform a morphological dilation on the cropped image.
6. Find contours using cv2.findContours
7. Plot the final image and offset the image by 200 pixels to show that the area of interest correctly matches up to the region where fine meshing is required.
'''

###
### Python code for implementing solution
###

import cv2, numpy as np, matplotlib.pyplot as plt, matplotlib

matplotlib.rcParams['image.cmap'] = 'gray'

imagePath = 'Quiz-1-Assets.png'

image = cv2.imread(imagePath,cv2.IMREAD_COLOR)
imageCopy = image.copy()
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold image

_, thImg = cv2.threshold(imageGray, 0, 255, cv2.THRESH_BINARY_INV)

# Crop image

thWC = thImg[200:500,:]

k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
m = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

thWC = cv2.dilate(thWC, m)

plt.title("Cropped area")
plt.xticks([]);plt.yticks([])
plt.imshow(thWC)
plt.show()
plt.close()

# Find contours

contours, hierarchy = cv2.findContours(thWC, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print("Number of contours: {}".format(len(contours)))

imageFinal = image.copy()

cv2.drawContours(imageFinal, contours, -1, (0,255,0), 3, offset=(0,200))

fig1 = plt.figure()
fig1.suptitle("Wind tunnel analysis")
plt.subplot(121);plt.imshow(image);plt.title("Original image")
plt.subplot(122);plt.imshow(imageFinal);plt.title("Final image")
plt.show()
plt.close()