#face recognition sample

import numpy as np
import time
import os
import sys
import cv2
import urllib.request

import ailia

print("downloading ...");

weight_path = "resnet50_scratch.caffemodel"
model_path = "resnet50_scratch.prototxt"

if not os.path.exists(model_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/ailia-models/vggface2/" + model_path, model_path)
if not os.path.exists(weight_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/ailia-models/vggface2/" + weight_path, weight_path)

print("loading ...");

img_a_path = 'couple_a.jpg'
img_b_path = 'couple_b.jpg'
img_c_path = 'couple_c.jpg'

env_id=0
env_id=ailia.get_gpu_environment_id()
net = ailia.Net(model_path,weight_path,env_id=env_id)

IMAGE_WIDTH=net.get_input_shape()[3]
IMAGE_HEIGHT=net.get_input_shape()[2]

print("inferencing ...");

features=[]
for img in [img_a_path,img_b_path,img_c_path]:
    input_img = cv2.imread(img)
    img = cv2.resize(input_img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    img = img[...,::-1]  #BGR 2 RGB
    data = np.array(img, dtype=np.float32)
    data.shape = (1,) + data.shape
    data = data - 128   # signed int8
    data = data.transpose((0, 3, 1, 2))
    pred = net.predict(data)
    blob=net.get_blob_data(net.find_blob_index_by_name("conv5_3"))
    features.append(blob)

threshold=1.00	#VGGFace2 predefined value 1〜1.24

def distance(feature1,feature2):
    norm1=np.sqrt(np.sum(np.abs(feature1**2)))
    norm2=np.sqrt(np.sum(np.abs(feature2**2)))
    dist=feature1/norm1-feature2/norm2
    l2_norm=np.sqrt(np.sum(np.abs(dist**2)))
    return l2_norm

print("image_a vs image_b = ",distance(features[0],features[1]))
if(distance(features[0],features[1])<threshold):
    print("same person")
else:
    print("not same person")

print("image_a vs image_c = ",distance(features[0],features[2]))
if(distance(features[0],features[2])<threshold):
    print("same person")
else:
    print("not same person")
