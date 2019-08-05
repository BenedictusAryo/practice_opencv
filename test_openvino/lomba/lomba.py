#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 20:14:14 2019

@author: benedict
"""

from openvino.inference_engine import IENetwork, IEPlugin
import cv2 as cv

#######################  Device  Initialization  ########################
#  Plugin initialization for specified device and load extensions library if specified

plugin = IEPlugin(device="CPU")
#    plugin = IEPlugin(device="MYRIAD")
#########################################################################

# prepare the model
## UBUNTU
model_xml = "/home/benedict/practice_opencv/test_openvino/lomba/age-gender-recognition-retail-0013.xml"
model_bin = "/home/benedict/practice_opencv/test_openvino/lomba/age-gender-recognition-retail-0013.bin"

# WINDOWS
#model_xml = "face-detection-retail-0004.xml"
#model_bin = "face-detection-retail-0004.bin"


            
#########################  Load Neural Network  #########################
#  Read in Graph file (IR)
net = IENetwork(model=model_xml, weights=model_bin) 

############# Load network to the plugin for cpu processing, ############ 
###################### no need for GPU or MYRIAD ########################

#########################  CHANGE THIS LINE FOR WINDOWS / UBUNTU ########

# Plugin UBUNTU :
plugin.add_cpu_extension("/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so")


# Plugin Windows
#plugin.add_cpu_extension(r"C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\inference_engine\bin\intel64\Release\cpu_extension_avx2.dll")

# Load the Network using Plugin Device
exec_net = plugin.load(network=net)
########################################################################

#########################  Obtain Input Tensor  ########################
#  Obtain and preprocess input tensor (image)
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
