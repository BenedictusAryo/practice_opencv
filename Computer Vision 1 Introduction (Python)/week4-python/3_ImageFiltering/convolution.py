import cv2
import numpy as np
from dataPath import DATA_PATH

filename = DATA_PATH+"images/sample.jpg"

image = cv2.imread(filename)

if image is None:  # Check if file is not present
    print("Could not open or find the image")

kernel_size = 5
# Create a 5*5 kernel with all elements equal to 1
kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / kernel_size**2

# Print Kernel
print (kernel)

result = cv2.filter2D(image, -1, kernel, (-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)

cv2.imshow("Original Image", image)
cv2.imshow("Convolution Result", result)
cv2.waitKey(0)

