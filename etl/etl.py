#ailia classifier api sample

import numpy as np
import time
import os
import sys
import cv2
import urllib.request
import codecs

import ailia

# settings
model_path = "etl_BINARY_squeezenet128_20.prototxt"
weight_path = "etl_BINARY_squeezenet128_20.caffemodel"

img_path = './font.png'

etl_word = codecs.open("etl_BINARY_squeezenet128_20.txt", 'r', 'utf-8').readlines()
print(etl_word)

# model download
print("downloading ...");

if not os.path.exists(model_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/ailia-models/etl/"+model_path,model_path)
if not os.path.exists(weight_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/ailia-models/etl/"+weight_path,weight_path)

# classifier initialize
print("loading ...");

env_id = ailia.get_gpu_environment_id()
classifier = ailia.Classifier(model_path, weight_path, env_id=env_id, format=ailia.NETWORK_IMAGE_FORMAT_GRAY, range=ailia.NETWORK_IMAGE_RANGE_U_FP32)

# load input image and convert to BGRA
print("inferencing ...");

img = cv2.imread( img_path, cv2.IMREAD_UNCHANGED )
if img.shape[2] == 3 :
    img = cv2.cvtColor( img, cv2.COLOR_BGR2BGRA )
elif img.shape[2] == 1 : 
    img = cv2.cvtColor( img, cv2.COLOR_GRAY2BGRA )

img = cv2.bitwise_not(img)

# compute
max_class_count = 3

cnt = 3
for i in range(cnt):
	start=int(round(time.time() * 1000))
	classifier.compute(img, max_class_count)
	end=int(round(time.time() * 1000))
	print("## ailia processing time , "+str(i)+" , "+str(end-start)+" ms")

# get result
count = classifier.get_class_count()

print("class_count=" + str(count))

for idx  in range(count) :
    # print result
    print("+ idx=" + str(idx))
    info = classifier.get_class(idx)
    print(info.category)
    print(len(etl_word))
    print("  category=" + str(info.category) + "[ " + etl_word[info.category] + " ]" )
    print("  prob=" + str(info.prob) )

