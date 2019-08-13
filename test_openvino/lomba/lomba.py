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
import argparse    

#######################  Create Argument Parser  ########################
parser = argparse.ArgumentParser(
    description="Smart DOOH using OpenVINO Face, Age & Gender Detection.")
parser.add_argument("-d", "--device", metavar='', default='CPU',
        help="Device to run inference: GPU, CPU or MYRIAD", type=str)
parser.add_argument("-c", "--camera", metavar='', default=0,
        help="Camera Device, default 0 for Webcam",type=int)
parser.add_argument("-s", "--sample", default=False,
        action='store_true', help="Inference using sample video")

args = parser.parse_args()

#######################  Device  Initialization  ########################
#  Plugin initialization for specified device and load extensions library if specified

device = args.device.upper()

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

## Model 1: Face Detection
# FACEDETECT_XML = "models/face-detection-retail-0004.xml"
# FACEDETECT_BIN = "models/face-detection-retail-0004.bin"
FACEDETECT_XML = "models/face-detection-adas-0001.xml"
FACEDETECT_BIN = "models/face-detection-adas-0001.bin"
## Model 2: Age Gender Recognition
AGEGENDER_XML = "models/age-gender-recognition-retail-0013_FP32.xml"
AGEGENDER_BIN = "models/age-gender-recognition-retail-0013_FP32.bin"

################  Create PostProcessing Inferece Function  ################

def gender_class(gender):
    """
    PostProcessing & Classify Output Gender into Male & Female
    """
    GENDER_LIST=['Female', 'Male']
    if gender[0,1,0,0] >= 0.70:
        return GENDER_LIST[1]
    else:
        return GENDER_LIST[0]
    

def age_class(age):
    """
    Classify Age whether it's below or above 30
    """
    return 'Below 30' if age <= 30 else 'Above 30'

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
net_facedetect, exec_facedetect = load_model(plugin, FACEDETECT_XML,FACEDETECT_BIN)
net_ageGender, exec_ageGender = load_model(plugin, AGEGENDER_XML, AGEGENDER_BIN)

###################  Obtain Input&Output Tensor  ######################
## Model 1
#  Define Input&Output Network dict keys
FACEDETECT_INPUTKEYS = 'data'
FACEDETECT_OUTPUTKEYS = 'detection_out'
#  Obtain image_count, channels, height and width
n_facedetect, c_facedetect, h_facedetect, w_facedetect = net_facedetect.inputs[FACEDETECT_INPUTKEYS].shape

## Model 2
#  Define Input&Output Network dict keys
AGEGENDER_INPUTKEYS = 'data'
AGE_OUTPUTKEYS = 'age_conv3'
GENDER_OUTPUTKEYS = 'prob'
#  Obtain image_count, channels, height and width
n_ageGender, c_ageGender, h_ageGender, w_ageGender = net_ageGender.inputs[AGEGENDER_INPUTKEYS].shape

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
if args.sample:
    input_stream = 'face-demographics-walking-and-pause.mp4'
else:
    input_stream = args.camera 

cap = cv.VideoCapture(input_stream)

#  If Video File, slow down the video playback based on FPS
if type(input_stream) is str:
    time.sleep(1/cap.get(cv.CAP_PROP_FPS))

while cv.waitKey(1) != ord('q'):
    if cap:
        hasFrame, image = cap.read()
            
    if not hasFrame:
        break

    ###################  Start  Inference Face Detection  ###################
    #  Start asynchronous inference and get inference result
    blob = image_preprocessing(image, n_facedetect, c_facedetect, h_facedetect, w_facedetect)
    req_handle = exec_facedetect.start_async(request_id=0, inputs={FACEDETECT_INPUTKEYS:blob})

    ######################## Get Inference Result  #########################
    status = req_handle.wait()
    res = req_handle.outputs[FACEDETECT_OUTPUTKEYS]

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
        bottomLeftCornerOfText = (xmin,ymin-10)
        fontScale = 1
        lineType = 2

        # Crop Face which having confidence > 90%
        if confidence > 0.9:
            ## Draw Boundingbox
            cv.rectangle(image, (xmin, ymin), (xmax, ymax), fontColor)
            
            ## Create CropFace to recognize ageGender
            crop_face = image[ymin:ymax, xmin:xmax]
            try:
            ## Infer to Gender & Age Model
                blob_ageGender = image_preprocessing(crop_face,n_ageGender,c_ageGender,h_ageGender,w_ageGender)
                req_handle_ageGender = exec_ageGender.start_async(request_id=0, inputs={AGEGENDER_INPUTKEYS:blob_ageGender})

                ## Get inference result
                # Age
                status = req_handle_ageGender.wait()
                age = req_handle_ageGender.outputs[AGE_OUTPUTKEYS]
                age = int(age[0,0,0,0] *100)
                # Gender
                gender = req_handle_ageGender.outputs[GENDER_OUTPUTKEYS]
                gender = gender[0,:,0,0].argmax()
                #print(age,GENDER_LIST[gender])

                # Put text of Age and Gender
                cv.putText(image,f"{gender_class(gender)}, {age_class(age)}",bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
            except:
                continue

    cv.namedWindow('AI_Vertising', cv.WINDOW_NORMAL)
    cv.moveWindow('AI_Vertising', 0,0)
    cv.resizeWindow('AI_Vertising',700,700)
    cv.imshow('AI_Vertising', image)

###############################  Clean  Up  ############################
del exec_facedetect
del exec_ageGender
del net_ageGender
del net_facedetect
del plugin
cap.release()
cv.destroyAllWindows()
########################################################################