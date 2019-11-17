#ailia classifier api sample

import numpy as np
import time
import os
import sys
import cv2
import urllib.request

import ailia

emotion_category=[
	"angry",
	"disgust",
	"fear",
	"happy",
	"sad",
	"surprise",
	"neutral"
]

gender_category=[
	"female","male"
]

print("downloading ...");

emotion_weight_path = "emotion_miniXception.caffemodel"
emotion_model_path = "emotion_miniXception.prototxt"

if not os.path.exists(emotion_model_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/ailia-models/face_classification/" + emotion_model_path, emotion_model_path)
if not os.path.exists(emotion_weight_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/ailia-models/face_classification/" + emotion_weight_path, emotion_weight_path)

gender_weight_path = "gender_miniXception.caffemodel"
gender_model_path = "gender_miniXception.prototxt"

if not os.path.exists(gender_model_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/ailia-models/face_classification/" + gender_model_path, gender_model_path)
if not os.path.exists(gender_weight_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/ailia-models/face_classification/" + gender_weight_path, gender_weight_path)

print("loading ...");

img_path = 'lenna.png'

# classifier initialize
env_id = ailia.get_gpu_environment_id()
emotion_classifier = ailia.Classifier(emotion_model_path, emotion_weight_path, env_id=env_id, format=ailia.NETWORK_IMAGE_FORMAT_GRAY, range=ailia.NETWORK_IMAGE_RANGE_S_FP32, channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST)
gender_classifier = ailia.Classifier(gender_model_path, gender_weight_path, env_id=env_id, format=ailia.NETWORK_IMAGE_FORMAT_GRAY, range=ailia.NETWORK_IMAGE_RANGE_S_FP32, channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST)

# load input image and convert to BGRA
img = cv2.imread( img_path, cv2.IMREAD_UNCHANGED )
if img.shape[2] == 3 :
    img = cv2.cvtColor( img, cv2.COLOR_BGR2BGRA )
elif img.shape[2] == 1 : 
    img = cv2.cvtColor( img, cv2.COLOR_GRAY2BGRA )

print("inferencing ...");

# compute emotion
emotion_max_class_count = 3
emotion_classifier.compute(img, emotion_max_class_count)
count = emotion_classifier.get_class_count()
print("emotion_class_count=" + str(count))
for idx  in range(count) :
    # print result
    print("+ idx=" + str(idx))
    info = emotion_classifier.get_class(idx)
    print("  category=" + str(info.category) + "[ " + emotion_category[info.category] + " ]" )
    print("  prob=" + str(info.prob) )
print("")

# compute gender
gender_max_class_count = 2
gender_classifier.compute(img, gender_max_class_count)
count = gender_classifier.get_class_count()
print("gender_class_count=" + str(count))
for idx  in range(count) :
    # print result
    print("+ idx=" + str(idx))
    info = gender_classifier.get_class(idx)
    print("  category=" + str(info.category) + "[ " + gender_category[info.category] + " ]" )
    print("  prob=" + str(info.prob) )
