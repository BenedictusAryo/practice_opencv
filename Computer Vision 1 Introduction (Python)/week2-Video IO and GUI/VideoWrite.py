# Import module
import cv2
from dataPath import DATA_PATH

cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video stream or file")

# Default resolutions of the frame are obtained.
# Convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.
outavi = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
outmp4 = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_width,frame_height))

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()

  if ret == True:

    # Write the frame into the file 'outputChaplin.mp4'
    outavi.write(frame)
    outmp4.write(frame)

    # Wait for 25 ms before moving on to the next frame
    # This will slow down the video
    cv2.imshow("Frame",frame)
    k = cv2.waitKey(25)
    if k == 27:
        break

  # Break the loop
  else:
    break

# When everything done, release the VideoCapture and VideoWriter objects
cap.release()
outavi.release()
outmp4.release()
