#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 20:14:14 2019

@author: benedict.aryo
"""
#######################################################################
######################  Library Initialization  #########################
#  Import Library being used in program
from openvino.inference_engine import IENetwork, IEPlugin
import cv2 as cv
import platform
import time

#######################################################################
#######################  Device  Initialization  ########################
#  Plugin initialization for specified device and load extensions library if specified

device = "CPU"

# Device Options = "CPU", "GPU", "MYRIAD"
plugin = IEPlugin(device=device)

# DETECT OS WINDOWS / UBUNTU  TO USE EXTENSION LIBRARY
# Plugin UBUNTU :
linux_cpu_plugin = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so"
# Plugin Windows
windows_cpu_plugin = r"C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\inference_engine\bin\intel64\Release\cpu_extension_avx2.dll"

if platform.system() == 'Windows':
    cpu_plugin = windows_cpu_plugin
else:
    cpu_plugin = linux_cpu_plugin

# Add Extension to Device Plugin
if device == "CPU":
    plugin.add_cpu_extension(cpu_plugin)
 
#################### no need for GPU or MYRIAD ########################
#######################################################################

#######################  MODEL INITIALIZATION  ########################
#  Prepare and load the models

## Model 1: Face Detection
model1_xml = "models/face-detection-retail-0004.xml"
model1_bin = "models/face-detection-retail-0004.bin"

## Model 2: Age Gender Recognition
model2_xml = "models/age-gender-recognition-retail-0013.xml"
model2_bin = "models/age-gender-recognition-retail-0013.bin"


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
net1, exec_net1 = load_model(plugin, model1_xml,model1_bin)
net2, exec_net2 = load_model(plugin, model2_xml, model2_bin)

###################  Obtain Input&Output Tensor  ######################
## Model 1
#  Obtain and preprocess input tensor (image)
input_net1 = 'data'
output_net1 = 'detection_out'
#  Obtain image_count, channels, height and width
n_model1, c_model1, h_model1, w_model1 = net1.inputs[input_net1].shape

## Model 2
#  Obtain and preprocess input tensor (image)
input_net2 = next(iter(net2.inputs))
output_net2 = next(iter(net2.outputs))
#  Obtain image_count, channels, height and width
n_model2, c_model2, h_model2, w_model2 = net2.inputs[input_net2].shape

def face_detect_preprocessing(n, c, h, w):
    """
    Image Preprocessing steps, to match image 
    with Input Neural nets
    
    N=1, Channel=3, Height=300, Width=300
    """
    blob = cv.resize(image, (w, h)) # Resize width & height
    blob = blob.transpose((2, 0, 1)) # Change data layout from HWC to CHW
    blob = blob.reshape((n, c, h, w))
    return blob

def age_gender_preprocessing(n, c, h, w):
    """
    Image Preprocessing steps, to match image 
    with Input Neural nets
    
    N=1, Channel=3, Height=62, Width=62
    """
    blob = cv.resize(image, (w, h)) # Resize width & height
    blob = blob.transpose((2, 0, 1)) # Change data layout from HWC to CHW
    blob = blob.reshape((n, c, h, w))
    return blob


#########################  Read Video Capture  ########################
#  Using OpenCV to read Video/Camera
vid_or_cam = 'face-demographics-walking-and-pause.mp4'
#  Use 0 for Webcam, 1 for Externaql Camera, or string with filepath for video
cap = cv.VideoCapture(vid_or_cam)

#  If Video File, slow down the video playback based on FPS
if type(vid_or_cam) is str:
    time.sleep(1/cap.get(cv.CAP_PROP_FPS))

while cv.waitKey(1) != ord('q'):
    if cap:
        hasFrame, image = cap.read()
            
    if not hasFrame:
        break

    ###################  Start  Inference Face Detection  ###################
    #  Start asynchronous inference and get inference result
    blob = face_detect_preprocessing(n_model1, c_model1, h_model1, w_model1)
    req_handle = exec_net1.start_async(request_id=0, inputs={input_net1:blob})

    ######################## Get Inference Result  #########################
    status = req_handle.wait()
    res = req_handle.outputs[output_net1]

    # Get Bounding Box Result
    for detection in res[0][0]:
        confidence = float(detection[2]) # Face detection Confidence
        # Obtain Bounding box coordinate
        xmin = int(detection[3] * image.shape[1])
        ymin = int(detection[4] * image.shape[0])
        xmax = int(detection[5] * image.shape[1])
        ymax = int(detection[6] * image.shape[0])

        # Crop Face which having confidence > 90%
        if confidence > 0.9:
            crop_face = image[ymin:ymax, xmin:xmax]

            # Infer to Gender & Age Model
            ###


            # Draw Boundingbox
            cv.rectangle(image, (xmin, ymin), (xmax, ymax), (0,0,255))
    
    cv.imshow('AI_Vertising', image)

###############################  Clean  Up  ############################
del exec_net1
del net1
del plugin
cap.release()
cv.destroyAllWindows()
########################################################################