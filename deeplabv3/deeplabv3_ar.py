import numpy as np
import time
import os
import cv2
import urllib.request
import sys

from PIL import Image

import ailia

OPT_MODEL=True
if OPT_MODEL:
    model_path = "deeplabv3.opt.onnx.prototxt"
    weight_path = "deeplabv3.opt.onnx"
else:
    model_path = "deeplabv3.onnx.prototxt"
    weight_path = "deeplabv3.onnx"

print("downloading ...");

if not os.path.exists(model_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/ailia-models/deeplabv3/"+model_path,model_path)
if not os.path.exists(weight_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/ailia-models/deeplabv3/"+weight_path,weight_path)

print("loading ...");

#env_id=ailia.ENVIRONMENT_AUTO
env_id=ailia.get_gpu_environment_id()
net = ailia.Net(model_path,weight_path,env_id=env_id)

print("inferencing ...");


ailia_input_width = net.get_input_shape()[3]
ailia_input_height = net.get_input_shape()[2]

LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

file_path = 'shibuya.mp4'
delay = 1
window_name = 'frame'

cap = cv2.VideoCapture(file_path)

if not cap.isOpened():
    sys.exit()

while True:
    ret, frame = cap.read()
    if ret:
        print("input frame : ")
        print(frame.shape)

        frame=frame[:,0:frame.shape[0],:]

        img_width=frame.shape[1]
        img_height=frame.shape[0]

        img = cv2.resize(frame,(ailia_input_width,ailia_input_height),interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img.shape = (1,) + img.shape
        img = img.transpose((0, 3, 1, 2))
        img = np.array(img)
        img = img.astype(np.float32)
        img = img / 127.0 - 1.0

        start=int(round(time.time() * 1000))
        output_img = net.predict(img)
        end=int(round(time.time() * 1000))
        print("## ailia processing time , "+str(end-start)+" ms")

        output_img[:,15,:,:]=output_img[:,15,:,:]   #person
        #output_img[:,16,:,:]=output_img[:,6,:,:]    #car
        output_img[:,17,:,:]=output_img[:,7,:,:]*0.8   #bus

        output_img = output_img[:,15:18,:,:]
        output_img = output_img.transpose((0,2,3,1))

        shape = output_img.shape
        output_img = output_img.reshape((shape[1],shape[2],shape[3]))
        output_img = output_img*255. / 21

        output_img[output_img<0] = 0
        output_img[output_img>255] = 255.      

        img_in = cv2.resize(frame,(img_width,img_height),interpolation=cv2.INTER_AREA)
        img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
        print("img in : ")
        print(img_in.shape)

        img2 = cv2.resize(output_img,(img_width,img_height),interpolation=cv2.INTER_CUBIC)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        print("img2 : ")
        print(img2.shape)

        th = 96
        img2[img2<th] = 0
        img2[img2>th] = 255.      

        img2 = img2 + img_in * 0.5

        img2[img2<0] = 0
        img2[img2>255] = 255.      
        numpyArray = np.array(img2) / 255

        cv2.imshow(window_name, numpyArray)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

cv2.destroyWindow(window_name)


