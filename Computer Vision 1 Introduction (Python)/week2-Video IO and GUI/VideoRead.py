# Import module
import cv2
from dataPath import DATA_PATH

cap = cv2.VideoCapture('chaplin.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    # Display the resulting frame
    cv2.imshow('Frame',frame)

    # Press esc on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == 27:
      break

  # Break the loop
  else:
    break
