#ailia detector api sample

import ailia

import numpy
import tempfile
import cv2
import os
import urllib.request

# settings
model_path = "yolov1-tiny.prototxt"
weight_path = "yolov1-tiny.caffemodel"
img_path = "./couple.jpg"

print("downloading ...");

if not os.path.exists(model_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/ailia-models/yolov1-tiny/"+model_path,model_path)
if not os.path.exists(weight_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/ailia-models/yolov1-tiny/"+weight_path,weight_path)

print("loading ...");

# detector initialize
env_id = ailia.get_gpu_environment_id()
categories = 20
detector = ailia.Detector(model_path, weight_path, categories, format=ailia.NETWORK_IMAGE_FORMAT_RGB, channel=ailia.NETWORK_IMAGE_CHANNEL_FIRST, range=ailia.NETWORK_IMAGE_RANGE_S_FP32, algorithm=ailia.DETECTOR_ALGORITHM_YOLOV1, env_id=env_id)

# load input image and convert to BGRA
img = cv2.imread( img_path, cv2.IMREAD_UNCHANGED )
if img.shape[2] == 3 :
    img = cv2.cvtColor( img, cv2.COLOR_BGR2BGRA )
elif img.shape[2] == 1 : 
    img = cv2.cvtColor( img, cv2.COLOR_GRAY2BGRA )

print( "img.shape=" + str(img.shape) )

work = img
w = img.shape[1]
h = img.shape[0]

print("inferencing ...");

# compute
threshold = 0.2
iou = 0.45
detector.compute(img, threshold, iou)

# category
voc_category=[
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

# get result
count = detector.get_object_count()

print("object_count=" + str(count))

for idx  in range(count) :
    # print result
    print("+ idx=" + str(idx))
    obj = detector.get_object(idx)
    print("  category=" + str(obj.category) + "[ " + voc_category[obj.category] + " ]" )
    print("  prob=" + str(obj.prob) )
    print("  x=" + str(obj.x) )
    print("  y=" + str(obj.y) )
    print("  w=" + str(obj.w) )
    print("  h=" + str(obj.h) )
    top_left = ( int(w*obj.x), int(h*obj.y) )
    bottom_right = ( int(w*(obj.x+obj.w)), int(h*(obj.y+obj.h)) )
    text_position = ( int(w*obj.x)+4, int(h*(obj.y+obj.h)-8) )

    # update image
    cv2.rectangle( work, top_left, bottom_right, (0, 0, 255, 255), 4)
    cv2.putText( work, voc_category[obj.category], text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255, 255), 1)

# save image
cv2.imwrite( "output.png", work)
