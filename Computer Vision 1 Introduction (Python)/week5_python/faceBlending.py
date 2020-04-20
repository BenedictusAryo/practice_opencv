# Standard imports
import cv2
import numpy as np 
from dataPath import DATA_PATH

# Read images
dst = cv2.imread(DATA_PATH + "images/trump.jpg")

src = cv2.imread(DATA_PATH + "images/obama.jpg")

src_mask = cv2.imread(DATA_PATH + "images/obama-mask.jpg", cv2.IMREAD_GRAYSCALE)

cv2.imshow("Trump",dst)
cv2.imshow("Obama",src)
cv2.imshow("Mask",src_mask)

cv2.waitKey(0)

alpha = cv2.cvtColor(src_mask.copy(), cv2.COLOR_GRAY2BGR)
alpha = alpha.astype(np.float32) / 255.0
output_blend = src * alpha + dst * (1 - alpha)
output_blend = output_blend.astype(np.uint8)
cv2.imshow("Simple Alpha Blending with Mask",output_blend)
cv2.waitKey(0)

# Find blob centroid
ret, src_mask_bin = cv2.threshold(src_mask, 128,255, cv2.THRESH_BINARY)
m = cv2.moments(src_mask_bin)
center = (int(m['m01']/m['m00']), int(m['m10']/m['m00']) )

output_clone = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)
cv2.imshow("Using Seamless Cloning",output_clone)
cv2.waitKey(0)

