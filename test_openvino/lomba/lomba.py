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

#######################  Model Initialization  ########################
#  Prepare and load the models

model_xml = "models/face-detection-retail-0004.xml"
model_bin = "models/face-detection-retail-0004.bin"

            
#########################  Load Neural Network  #########################
#  Read in Graph file (IR)
net = IENetwork(model=model_xml, weights=model_bin) 

# Load the Network using Plugin Device
exec_net = plugin.load(network=net)
############# Load network to the plugin for cpu processing, ############
########################################################################

#########################  Obtain Input Tensor  ########################
#  Obtain and preprocess input tensor (image)
input_layer = next(iter(net.inputs))
out_layer = next(iter(net.outputs))
#  Obtain image_count, channels, height and width
n, c, h, w = net.inputs[input_layer].shape

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


#########################  Read Video Capture  ########################
#  Using OpenCV to read Video/Camera
vid_or_cam = 0 #'face-demographics-walking-and-pause.mp4'
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

    ########################################################################
    
    ##########################  Start  Inference  ##########################
    #  Start asynchronous inference and get inference result
    blob = face_detect_preprocessing(n, c, h, w)
    req_handle = exec_net.start_async(request_id=0, inputs={input_layer:blob})

    ######################## Get Inference Result  #########################
    status = req_handle.wait()
    res = req_handle.outputs[out_layer]

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
    
    cv.imshow('AI_Vetising', image)

###############################  Clean  Up  ############################
del exec_net
del net
del plugin
cap.release()
cv.destroyAllWindows()
########################################################################