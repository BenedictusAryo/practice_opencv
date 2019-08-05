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
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
