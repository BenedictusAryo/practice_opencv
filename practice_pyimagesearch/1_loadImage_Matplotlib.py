"""
Load Images, Show and saving using Matplotlib
"""

# Import Library
from __future__ import print_function
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Create arguments that takes input image PATH
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
        help = "Path to the image")
args = vars(ap.parse_args())

# Read and Show the image 
image = mpimg.imread(args["image"])
print("Width : {} pixels".format(image.shape[1]))
print("Height : {} pixels".format(image.shape[0]))
print("Channels : {} ".format(image.shape[2]))

# plt.axis("off")
plt.imshow(image)
plt.show()


