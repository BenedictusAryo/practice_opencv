import numpy as np
import cv2

# Sample 2x2 matrix of type uint8
a = np.array([[100, 110], 
              [120, 130]], dtype='uint8')
print(a)

# Add 130 so that the last element encounters overflow
print(a + 130)

print(a - 130)
print(a + (-130))

#Solution 1 : Use opencv functions
print(cv2.add(a,130))

#Solution 2a: Convert to int32/int64
a_int32 = np.int32(a)
b = a_int32+130
print(b)

print(b.clip(0,255))
b_uint8 = np.uint8(b)
b_uint8

#Solution 2b: Convert to normalized float32/float64 
a_float32 = np.float32(a)/255
b = a_float32 + 130/255
print(b)

c = b*255
print("Output = \n{}".format(c))
print("Clipped output= \n{}".format(c.clip(0,255)))
b_uint8 = np.uint8(c.clip(0,255))
print("uint8 output = \n{}".format(b_uint8))

