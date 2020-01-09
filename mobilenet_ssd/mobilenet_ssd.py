import sys
import os
import urllib.request
import time

import numpy as np
import cv2

import ailia
from  utils import save_result


img_path = "couple.jpg"
save_path = "annotated.jpg"


# model loading part
if len(sys.argv)<2:
    net_type="mb2-ssd-lite"
else:
    net_type = sys.argv[1]

model_lists = ['mb1-ssd', 'mb2-ssd-lite']
if net_type in model_lists:
    model_path = net_type + '.onnx.prototxt'
    weight_path = net_type + '.onnx'
else:
    print("The net type is wrong.")
    print("It should be mb1-ssd or mb2-ssd-lite.")
    sys.exit(1)

# model download
rmt_ckpt = "https://storage.googleapis.com/ailia-models/mobilenet_ssd/"
if not os.path.exists(model_path):
    urllib.request.urlretrieve(rmt_ckpt + model_path, model_path)
if not os.path.exists(weight_path):
    urllib.request.urlretrieve(rmt_ckpt + weight_path, weight_path)


# image loading part
img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (300, 300))
org_img = img
# [x, y, channel] --> [1, channel, x, y]
img = img / 255.0
img = np.expand_dims(np.rollaxis(img, 2, 0), axis=0) 


# model initialize
env_id = ailia.get_gpu_environment_id()
net = ailia.Net(model_path, weight_path, env_id=env_id)

for i in range(1):
    start = int(round(time.time() * 1000))
    # pred = net.predict(img)
    input_blobs = net.get_input_blob_list()
    net.set_input_blob_data(img, input_blobs[0])
    net.update()
    scores, boxes = net.get_results()
    end = int(round(time.time() * 1000))
    print("ailia processing time {} ms".format(end-start))

save_result(org_img, scores, boxes, save_path)
print('Successfully finished !')
