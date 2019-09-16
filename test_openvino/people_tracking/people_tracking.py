# People Tracking and Counting using OpenVino

"""
Created on Mon Sep  2 15:50:14 2019

@author: benedict.aryo
"""
#######################################################################
######################  Library Initialization  #########################
#  Import Library being used in program
from openvino.inference_engine import IENetwork, IEPlugin
import cv2 as cv
import platform
import time

#######################  Device  Initialization  ########################
#  Plugin initialization for specified device and load extensions library if specified

device = "CPU"

# Device Options = "CPU", "GPU", "MYRIAD"
plugin = IEPlugin(device=device)

# DETECT OS WINDOWS / UBUNTU  TO USE EXTENSION LIBRARY
# Plugin UBUNTU :
LINUX_CPU_PLUGIN = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so"
# Plugin Windows
WINDOWS_CPU_PLUGIN = r"C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\inference_engine\bin\intel64\Release\cpu_extension_avx2.dll"

if platform.system() == 'Windows':
    cpu_plugin = WINDOWS_CPU_PLUGIN
else:
    cpu_plugin = LINUX_CPU_PLUGIN

# Add Extension to Device Plugin
if device == "CPU":
    plugin.add_cpu_extension(cpu_plugin)
 
#################### no need for GPU or MYRIAD ########################
#######################################################################

#######################  MODEL INITIALIZATION  ########################
#  Prepare and load the models

## Model 1: People Detection
DETECTION_XML = "models/person-detection-retail-0013.xml"
DETECTION_BIN = "models/person-detection-retail-0013.bin"

#########################  Load Neural Network  #########################
def load_model(plugin, model, weights):
    """
    Load OpenVino IR Models

    Input:
    Plugin = Hardware Accelerator
    Model = model_xml file 
    Weights = model_bin file
    
    Output:
    execution network (exec_net)
    """
    #  Read in Graph file (IR) to create network
    net = IENetwork(model, weights) 
    # Load the Network using Plugin Device
    exec_net = plugin.load(network=net)
    return net, exec_net

####################  Create Execution Network  #######################
net_detect, exec_detect = load_model(plugin, DETECTION_XML, DETECTION_BIN)

###################  Obtain Input&Output Tensor  ######################
## Model 1
#  Define Input&Output Network dict keys
DETECTION_INPUTKEYS = 'data'
DETECTION_OUTPUTKEYS = 'detection_out'

#  Obtain image_count, channels, height and width
n_detect, c_detect, h_detect, w_detect = net_detect.inputs[DETECTION_INPUTKEYS].shape

# Image Preprocessing before goes to input Neural Network
def image_preprocessing(image,n, c, h, w):
    """
    Image Preprocessing steps, to match image 
    with Input Neural nets
    
    Image,
    N, Channel, Height, Width
    """
    blob = cv.resize(image, (w, h)) # Resize width & height
    blob = blob.transpose((2, 0, 1)) # Change data layout from HWC to CHW
    blob = blob.reshape((n, c, h, w))
    return blob

#########################  READ VIDEO CAPTURE  ########################
#  Using OpenCV to read Video/Camera
#  Use 0 for Webcam, 1 for External Camera, or string with filepath for video
cap = cv.VideoCapture('models/people-detection.mp4')

while cv.waitKey(1) != ord('q'):
    if cap:
        hasFrame, image = cap.read()
            
    if not hasFrame:
        break

    ###################  Start  Inference Face Detection  ###################
    #  Start asynchronous inference and get inference result
    blob = image_preprocessing(image, n_detect, c_detect, h_detect, w_detect)
    req_handle = exec_detect.start_async(request_id=0, inputs={DETECTION_INPUTKEYS:blob})

    ######################## Get Inference Result  #########################
    status = req_handle.wait()
    res = req_handle.outputs[DETECTION_OUTPUTKEYS]

    # Get Bounding Box Result
    for detection in res[0][0]:
        confidence = float(detection[2]) # Face detection Confidence
        # Obtain Bounding box coordinate, +-10 just for padding
        xmin = int(detection[3] * image.shape[1] -10)
        ymin = int(detection[4] * image.shape[0] -10)
        xmax = int(detection[5] * image.shape[1] +10)
        ymax = int(detection[6] * image.shape[0] +10)

        # OpenCV Drawing Set Up
        font = cv.FONT_HERSHEY_SIMPLEX
        fontColor = (0,0,255)
        boxColor = (255,0,255)
        bottomLeftCornerOfText = (xmin,ymin-10)
        fontScale = 1
        lineType = 2

        # Crop Face which having confidence > 90%
        if confidence > 0.9:
            ## Draw Boundingbox
            cv.rectangle(image, (xmin, ymin), (xmax, ymax), fontColor)
            

    cv.namedWindow('Person Detection', cv.WINDOW_NORMAL)
    cv.moveWindow('Person Detection', 0,0)
    cv.imshow('Person Detection', image)

###############################  Clean  Up  ############################
del exec_detect
del net_detect
del plugin
cap.release()
cv.destroyAllWindows()
########################################################################