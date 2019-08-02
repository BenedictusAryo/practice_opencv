import sys
import cv2
from openvino.inference_engine import IENetwork, IEPlugin


model_xml = 'face-detection-adas-0001.xml'
model_bin = 'face-detection-adas-0001.bin'

net = IENetwork(model=model_xml, weights=model_bin)

plugin = IEPlugin(device='CPU')
cpu_dll_path = "C:\Program Files (x86)\IntelSWTools\openvino_2019.1.087\deployment_tools\inference_engine\bin\intel64\Release\cpu_extension_avx2.dll"

plugin.add_cpu_extension(r"C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\inference_engine\bin\intel64\Release\cpu_extension_avx2.dll")
exec_net = plugin.load(network=net)

input_blob = next(iter(net.inputs)) #Input_blob = 'Data'
out_blob = next(iter(net.outputs)) #Out_blob = 'Detection_out'
model_n, model_c, model_h, model_w = net.inputs[input_blob].shape #Model_n, Model_c, Model_h, Model_w = 1, 3, 384, 672

del net

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cap_w = cap.get(3)
    cap_h = cap.get(4)
    in_frame = cv2.resize(frame, (model_w,model_h))
    in_frame = in_frame.transpose((2, 0, 1)) # Change data layout from HWC to CHW
    in_frame = in_frame.reshape((model_n, model_c, model_h, model_w))
    
    
    exec_net.start_async(request_id=0, inputs={input_blob:in_frame})
    
    if exec_net.requests[0].wait(-1) == 0:
        res = exec_net.requests[0].outputs[out_blob]
        
        for obj in res[0][0]:
            if obj[2] > 0.5:
                xmin = int(obj[3] * cap_w)
                ymin = int(obj[4] * cap_h)
                xmax = int(obj[5] * cap_w)
                ymax = int(obj[6] * cap_h)
                class_id = int(obj[1])
                
                # Draw box and label\class_id
                color = (255,0,0)
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)
                cv2.putText(frame, str(class_id)+' '+str(round(obj[2] * 100, 1))+' %',
                            (xmin,ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
    
    cv2.imshow("Detection Results", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
    
cv2.destroyAllWindows()
del exec_net
del plugin