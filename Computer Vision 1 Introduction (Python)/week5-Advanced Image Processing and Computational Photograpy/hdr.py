import cv2
import numpy as np
from dataPath import DATA_PATH

def readImagesAndTimes():
  # List of exposure times
  times = np.array([ 1/30.0, 0.25, 2.5, 15.0 ], dtype=np.float32)

  # List of image filenames
  filenames = ["img_0.033.jpg", "img_0.25.jpg", "img_2.5.jpg", "img_15.jpg"]
  images = []
  for filename in filenames:
    im = cv2.imread(DATA_PATH + "images/" + filename)
    images.append(im)

  return images, times

images, times = readImagesAndTimes()

# Align input images
alignMTB = cv2.createAlignMTB()
alignMTB.process(images, images)

# Obtain Camera Response Function (CRF)
calibrateDebevec = cv2.createCalibrateDebevec()
responseDebevec = calibrateDebevec.process(images, times)

# Merge images into an HDR linear image
mergeDebevec = cv2.createMergeDebevec()
hdrDebevec = mergeDebevec.process(images, times, responseDebevec)
# Save HDR image.
cv2.imwrite("hdrDebevec.hdr", hdrDebevec)


# Tonemap using Drago's method to obtain 24-bit color image
tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
ldrDrago = tonemapDrago.process(hdrDebevec)
ldrDrago = 3 * ldrDrago
cv2.imshow("Drago Tonemap", ldrDrago)
cv2.waitKey(0)

# Tonemap using Reinhard's method to obtain 24-bit color image
tonemapReinhard = cv2.createTonemapReinhard(1.5, 0,0,0)
ldrReinhard = tonemapReinhard.process(hdrDebevec)
cv2.imshow("Reinhard Tonemap", ldrReinhard)
cv2.waitKey(0)

# Tonemap using Mantiuk's method to obtain 24-bit color image
tonemapMantiuk = cv2.createTonemapMantiuk(2.2,0.85, 1.2)
ldrMantiuk = tonemapMantiuk.process(hdrDebevec)
ldrMantiuk = np.clip(3 * ldrMantiuk,0,1)
cv2.imshow("Mantiuk Tonemap", ldrMantiuk)
cv2.waitKey(0)
