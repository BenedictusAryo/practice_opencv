#!/usr/bin/env python3
import cv2
import numpy as np
import keras
from keras.utils.generic_utils import CustomObjectScope
from flask import Flask, abort, request 
import json
import base64
import io
from imageio import imread
import tensorflow as tf
from keras import backend as K
from flask import jsonify, make_response
import platform
import time
from openvino.inference_engine import IENetwork, IEPlugin


app = Flask(__name__) #create the Flask app

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

## Model 1: Face Detection
FACEDETECT_XML = "models/face-detection-retail-0004.xml"
FACEDETECT_BIN = "models/face-detection-retail-0004.bin"

## Model 2: Age Gender Recognition
AGEGENDER_XML = "models/age-gender-recognition-retail-0013.xml"
AGEGENDER_BIN = "models/age-gender-recognition-retail-0013.bin"

GENDER_LIST=['Female', 'Male']

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
    blob = cv2.resize(image, (w, h)) # Resize width & height
    blob = blob.transpose((2, 0, 1)) # Change data layout from HWC to CHW
    blob = blob.reshape((n, c, h, w))
    return blob

    
@app.route('/', methods=['POST'])
def recognize():
    data = request.form.get('img')
    image = imread(io.BytesIO(base64.b64decode(str(data))))
    image = cv2.resize(image, (224, 224))
    #image = cv2.
    #cv2.imwrite('test.jpg', image)

    g = 'Unknown'
    a = 0
    try:
        blob_ageGender = image_preprocessing(image,n_ageGender,c_ageGender,h_ageGender,w_ageGender)
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
        g = GENDER_LIST[gender]
        a = age_class(age)
    except:
        print('err')
    
    res = {"age":a,"gender":g}
    #return json.dumps(data)
    #response = app.response_class(response=json.dumps(res), status=200, mimetype='application/json')
    #print(res)
    return make_response(jsonify(res), 200)

    

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000) #run app in debug mode on port 5000