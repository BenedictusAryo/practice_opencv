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
