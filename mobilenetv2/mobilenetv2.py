#ailia classifier api sample

import numpy as np
import time
import os
import sys
import cv2
import urllib.request

import ailia
import mobilenetv2_labels

# settings
model_path = "mobilenetv2_1.0.onnx.prototxt"
weight_path = "mobilenetv2_1.0.onnx"
img_path = './clock.jpg'

# model download
print("downloading ...");

if not os.path.exists(model_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/ailia-models/mobilenetv2/"+model_path,model_path)
if not os.path.exists(weight_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/ailia-models/mobilenetv2/"+weight_path,weight_path)

# classifier initialize
print("loading ...");

env_id = ailia.get_gpu_environment_id()
classifier = ailia.Classifier(model_path, weight_path, env_id=env_id, format=ailia.NETWORK_IMAGE_FORMAT_RGB, range=ailia.NETWORK_IMAGE_RANGE_U_FP32)

# load input image and convert to BGRA
print("inferencing ...");

img = cv2.imread( img_path, cv2.IMREAD_UNCHANGED )
if img.shape[2] == 3 :
    img = cv2.cvtColor( img, cv2.COLOR_BGR2BGRA )
elif img.shape[2] == 1 : 
    img = cv2.cvtColor( img, cv2.COLOR_GRAY2BGRA )

# compute
max_class_count = 3
classifier.compute(img, max_class_count)

# get result
count = classifier.get_class_count()

print("class_count=" + str(count))

for idx  in range(count) :
    # print result
    print("+ idx=" + str(idx))
    info = classifier.get_class(idx)
    print("  category=" + str(info.category) + "[ " + mobilenetv2_labels.imagenet_category[info.category] + " ]" )
    print("  prob=" + str(info.prob) )

