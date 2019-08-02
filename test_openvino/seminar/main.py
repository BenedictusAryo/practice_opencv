#!/usr/bin/env python3
# (C) Copyright 2018, Intel Corporation
# SPDX-License-Identifier: MIT
from openvino.inference_engine import IENetwork, IEPlugin
from os.path import isfile, join
from scipy import spatial

import logging as log
import cv2 as cv
import os
import sys

def main():
    #######################  Device  Initialization  ########################
    #  Plugin initialization for specified device and load extensions library if specified
    
    plugin = IEPlugin(device="CPU")
#    plugin = IEPlugin(device="MYRIAD")
    #########################################################################

    # prepare the model
    #model_xml = "/home/intel/intel_models/face-detection-retail-0004/face-detection-retail-0004.xml"
    #model_bin = "/home/intel/intel_models/face-detection-retail-0004/face-detection-retail-0004.bin"
    model_xml = "face-detection-retail-0004.xml"
    model_bin = "face-detection-retail-0004.bin"
    

                
    #########################  Load Neural Network  #########################
    #  Read in Graph file (IR)
    net = IENetwork(model=model_xml, weights=model_bin) 
	# Load network to the plugin for cpu processing, no need for GPU or MYRIAD
	
	#plugin.add_cpu_extension("/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_avx2.so")
    plugin.add_cpu_extension(r"C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\inference_engine\bin\intel64\Release\cpu_extension_avx2.dll")
    exec_net = plugin.load(network=net)
    ########################################################################
    
    #########################  Obtain Input Tensor  ########################
    #  Obtain and preprocess input tensor (image)
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    
    
    #  Read and pre-process input image  
#    cap = cv.VideoCapture("/home/intel/sample-videos/face-demographics-walking-and-pause.mp4")
    cap = cv.VideoCapture(0)
    
    while cv.waitKey(1) != ord('q'):
        if cap:
            hasFrame, image = cap.read()  
        if not hasFrame:
            break
        #  Preprocessing is neural network dependent maybe we don't show this
#         print(net.inputs[input_blob].shape)
        n, c, h, w = net.inputs[input_blob].shape
        blob = cv.resize(image, (w, h))
        blob = blob.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        blob= blob.reshape((n, c, h, w))
        ########################################################################
        
        ##########################  Start  Inference  ##########################
        #  Start asynchronous inference and get inference result
        req_handle = exec_net.start_async(request_id=0, inputs={input_blob: blob})
        ########################################################################
        
        ######################## Get Inference Result  #########################
        status = req_handle.wait()
        res = req_handle.outputs[out_blob]
        
		# Do something with the results... (like print top 5)
        
        for detection in res[0][0]:
            print(detection)
            confidence = float(detection[2])
            xmin = int(detection[3] * image.shape[1])
            ymin = int(detection[4] * image.shape[0])
            xmax = int(detection[5] * image.shape[1])
            ymax = int(detection[6] * image.shape[0])
            nama = ""
    
            if confidence > 0.9:            
                crop_face = image[ymin:ymax, xmin:xmax]                        
#                 nama = fr.findPerson(crop_face)
                fontColor          = (0,255,0)                
                if nama == "":
                    fontColor          = (0,0,255)
                    nama = "unknown"
                
                font                   = cv.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (xmin,ymin)
                fontScale              = 1
                lineType               = 2
                cv.rectangle(image, (xmin, ymin), (xmax, ymax), fontColor)
                if isinstance(nama, str):# and matched_face/total_face > 0.5:
                    cv.putText(image, nama, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
        cv.imshow('OpenVINO face detection', image)
        # end of cam 01
    
    
    ###############################  Clean  Up  ############################
    del exec_net
    del net
    del plugin
    cap.release()
    cv.destroyAllWindows()
    ########################################################################
    

if __name__ == '__main__':
    sys.exit(main() or 0)
