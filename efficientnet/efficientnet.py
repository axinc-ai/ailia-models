import os
import time
import sys
import cv2

import ailia
import efficientnet_labels

#settings
model_path = "efficientnet-b7.onnx.prototxt"
weight_path = "efficientnet-b7.onnx"

if len(sys.argv) == 2:
    img_path = sys.argv[1]
else:
    img_path = "teddy_bear.jpg"

#load and resize input image
net = ailia.Net(model_path,weight_path)
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
temp = str(net.get_input_shape())
table = temp.maketrans({
    ' ':'',
    '(':'',
    ')':''
})
temp = temp.translate(table).split(',')
img = cv2.resize(img, (int(temp[2]), int(temp[3])))
del temp;del table

#convert to BGRA
if img.shape[2] == 3 :
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
elif img.shape[2] == 1 :
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)

#classifier initialize
env_id = ailia.get_gpu_environment_id()
classifier = ailia.Classifier(model_path, weight_path, format=ailia.NETWORK_IMAGE_FORMAT_RGB, range=ailia.NETWORK_IMAGE_RANGE_S_FP32)

# compute
max_class_count = 3
cnt = 3
for i in range(cnt):
	start = int(round(time.time() * 1000))
	classifier.compute(img, max_class_count)
	end = int(round(time.time() * 1000))
	print("## ailia processing time , " + str(i) + " , " + str(end - start) + " ms")

# get result
count = classifier.get_class_count()
print("class_count=" + str(count))
for idx  in range(count) :
    # print result
    print("+ idx=" + str(idx))
    info = classifier.get_class(idx)
    print("  category=" + str(info.category) + "[ " + efficientnet_labels.imagenet_category[info.category] + " ]" )
    print("  prob=" + str(info.prob) )
