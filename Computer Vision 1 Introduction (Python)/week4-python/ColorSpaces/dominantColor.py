import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
matplotlib.rcParams['image.interpolation']='bilinear'

from dataPath import DATA_PATH

filename = DATA_PATH+"images/jersey.jpg"

img = cv2.imread(filename)
cv2.imshow("Messi",img)
cv2.waitKey(0)

# Convert to HSV color space
hsvImage = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

# Split the channels
H, S, V = cv2.split(hsvImage)
print(H)
print(H.shape)

# Remove unsaturated (white/gray) pixels
H_array = H[S > 10].flatten()

print(H_array)
print(H_array.shape)

plt.figure(figsize=[20,10])
plt.subplot(121);plt.imshow(img[...,::-1]);plt.title("Image");plt.axis('off')
plt.subplot(122);plt.hist(H_array, bins=180, color='r');plt.title("Histogram")
plt.show()

cv2.destroyAllWindows()
