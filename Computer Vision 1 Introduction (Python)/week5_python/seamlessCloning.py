import cv2
import numpy as np 
from dataPath import DATA_PATH

# Read images
src = cv2.imread(DATA_PATH + "images/airplane.jpg")
dst = cv2.imread(DATA_PATH + "images/sky.jpg")

# Create a rough mask around the airplane.
src_mask = np.zeros(src.shape, src.dtype)
poly = np.array([ [4,80], [30,54], [151,63], [254,37], [298,90], [272,134], [43,122] ], np.int32)
src_mask = cv2.fillPoly(src_mask, [poly], (255, 255, 255))

# This is where the CENTER of the airplane will be placed
center = (800,100)

# Clone seamlessly.
output = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)

cv2.imshow("Output",output)
cv2.waitKey(0)

# Read images : src image will be cloned into dst
im = cv2.imread(DATA_PATH + "images/wood-texture.jpg")
obj= cv2.imread(DATA_PATH + "images/iloveyouticket.jpg")

# Create an all white mask
mask = 255 * np.ones(obj.shape, obj.dtype)

# The location of the center of the src in the dst
width, height, channels = im.shape
center = (height//2, width//2)

# Seamlessly clone src into dst and put the results in output
normal_clone = cv2.seamlessClone(obj, im, mask, center, cv2.NORMAL_CLONE)
mixed_clone = cv2.seamlessClone(obj, im, mask, center, cv2.MIXED_CLONE)

cv2.imshow("Normal Clone Result", normal_clone)
cv2.imshow("Mixed Clone Result", mixed_clone)
cv2.waitKey(0)
