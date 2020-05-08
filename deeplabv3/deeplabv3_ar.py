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
ar = cv2.imread('ar.png',cv2.IMREAD_UNCHANGED)

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

def shear_X(image, shear):
    h, w = image.shape[:2]
    src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
    dest = src.copy()
    dest[:,0] += (shear / h * (h - src[:,1])).astype(np.float32)
    #affine = cv2.getAffineTransform(src, dest)

    a = 0.3
    pts1 = np.float32([[0,0],[w,0],[w,h],[0,h]])
    pts2 = np.float32([[w*(a),0],[w*(1.0-a),0],[w,h],[0,h]])
    affine = cv2.getPerspectiveTransform(pts1,pts2)

    #affine = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1.0)
    #affine = affine * share
    
    return cv2.warpPerspective(image,affine,(w,h))
    #return cv2.warpAffine(image, affine, (w, h))

print(ar.shape)
ar = shear_X(ar, -20)
print(ar.shape)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
idx = 0#fps*32*60

while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    current_pos = str(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    print(idx,current_pos)
    idx = idx + fps
    ret, frame = cap.read()
    if ret:
        print("input frame : ")
        print(frame.shape)

        frame=frame[:,0:frame.shape[0],:]

        img_width=frame.shape[1]
        img_height=frame.shape[0]

        img = cv2.resize(frame,(ailia_input_width,ailia_input_height),interpolation=cv2.INTER_AREA)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img.shape = (1,) + img.shape
        img = img.transpose((0, 3, 1, 2))
        img = np.array(img)
        img = img.astype(np.float32)
        img = img / 127.0 - 1.0

        start=int(round(time.time() * 1000))
        output_img = net.predict(img)
        #output_img=np.zeros((1,21,64,64),dtype=np.float32)
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

        ar_width = 700
        ar_height = int(ar_width/2)
        px = 50
        py = 100
        ar_logo = cv2.resize(ar,(ar_width,ar_height),interpolation=cv2.INTER_AREA)
        ar_resize = cv2.resize(np.zeros((1, 1, 4), np.uint8), (img_width, img_height))
        ar_resize[int((img_height-ar_height)/2)+py:int((img_height-ar_height)/2+ar_height)+py,int((img_width-ar_width)/2)+px:int((img_width-ar_width)/2+ar_width)+px,:] = ar_logo

        img_in = cv2.resize(frame,(img_width,img_height),interpolation=cv2.INTER_AREA)
        #img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
        print("img in : ")
        print(img_in.shape)

        img2 = cv2.resize(output_img,(img_width,img_height),interpolation=cv2.INTER_CUBIC)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        print("img2 : ")
        print(img2.shape)

        th = 96
        img2[img2<=th] = 0
        img2[img2>th] = 255.      

        print("ar : ")
        print(ar_resize.shape)

        AR_MODE = False
        if AR_MODE:
            ar_resize[img2[:,:,0]>th]=0
            ar_resize[img2[:,:,1]>th]=0
            ar_resize[img2[:,:,2]>th]=0
            mask=ar_resize[:,:,3]/512.0
            mask=mask.reshape((img_width,img_height,1))
            a=ar_resize[:,:,0:3] * mask
            b=img_in * (1.0 - mask)
            img2=a+b
        else:
            img2 = img2 + img_in * 0.5# + ar_resize * 0.5
            #img2 = img_in + ar_resize * 1.0
        
        #img2 = img_in

        img2[img2<0] = 0
        img2[img2>255] = 255.      
        numpyArray = np.array(img2) / 255

        cv2.imshow(window_name, numpyArray)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

cv2.destroyWindow(window_name)


