#ailia classifier api sample

import numpy as np
import time
import os
import sys
import cv2

import ailia
import resnet50_labels

# settings
model_path = "./resnet50.onnx.prototxt"
weight_path = "./resnet50.onnx"
img_path = '../images/pizza.jpg'

# classifier initialize
env_id = ailia.get_gpu_environment_id()
classifier = ailia.Classifier(model_path, weight_path, env_id=env_id, format=ailia.NETWORK_IMAGE_FORMAT_RGB, range=ailia.NETWORK_IMAGE_RANGE_S_INT8)

# load input image and convert to BGRA
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
    print("  category=" + str(info.category) + "[ " + resnet50_labels.imagenet_category[info.category] + " ]" )
    print("  prob=" + str(info.prob) )

