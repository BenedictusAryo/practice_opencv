################################################
#
# Write Video file from Webcam or another Video
#
################################################

# Import module
import cv2
import platform  # platform used to specify codec based on OS

# Read webcam using VideoCapture
# and assign cap object
webcam_device = 0
cap = cv2.VideoCapture(webcam_device)

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Obtain default resolutions of the frame
# Convert the resolutions from float to integer
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object.
# WINDoWS -- *'DIVX'
# Linux -- *'XVID'
if platform.system() == 'Windows':
    os_codec = 'DIVX'
else:
    os_codec = 'XVID'
outavi = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(
    'M', 'J', 'P', 'G'), 20, (frame_width, frame_height))
outmp4 = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(
    *os_codec), 20, (frame_width, frame_height))

# Read until video is completed or stopped
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Write the frame into the file
    outavi.write(frame)
    outmp4.write(frame)

    # Showing video
    cv2.imshow('Video Stream', frame)

    # Press 'ESC' to stop the video
    if cv2.waitKey(1) & 0xFF == 27:
        break

# When everything done, release the VideoCapture and VideoWriter objects
cap.release()
outavi.release()
outmp4.release()
