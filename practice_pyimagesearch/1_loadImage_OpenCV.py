"""
Load Images, Show and saving using OpenCV
"""

# Import Library
from __future__ import print_function
import argparse
import cv2

# Create arguments that takes input image PATH
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
        help = "Path to the image")
args = vars(ap.parse_args())

# Read and Show the image 
image = cv2.imread(args["image"])
print("Width : {} pixels".format(image.shape[1]))
print("Height : {} pixels".format(image.shape[0]))
print("Channels : {} ".format(image.shape[2]))

cv2.imshow("Image", image)
cv2.waitKey(0)

# Save image to new location
cv2.imwrite("output/newimage.jpg", image)
