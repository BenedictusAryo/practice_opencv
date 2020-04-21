###############################################
#
# Read Video file and Display using OpenCV GUI
#
###############################################

# Import module
import cv2

# Read video file using VideoCapture
# and assign cap object
video_file = 'chaplin.mp4'
cap = cv2.VideoCapture(video_file)

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream of file")

# This code initiates an infinite loop (to be broken later by a break statement),
# where we have ret and frame being defined as the cap.read().
# Basically, ret is a boolean regarding whether or not
# there was a return at all, at the frame is each frame that is returned.
# If there is no frame, you wont get an error, you will get None. [1]
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        # Display the resulting frame
        cv2.imshow(f'Video: {video_file}', frame)

        # can also Press esc on keyboard to exit [2]
        if cv2.waitKey(25) & 0xFF == 27:  # 27 is ascii code for ESC key
            break

    # Break the loop if ret = False, means, video is over
    else:
        break


# Resource
# [1] https://pythonprogramming.net/loading-video-python-opencv-tutorial/
# [2] https://theasciicode.com.ar/ascii-control-characters/escape-ascii-code-27.html
