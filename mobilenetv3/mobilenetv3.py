import numpy as np
import time
import os
import cv2
import urllib.request
import argparse

import ailia
import mobilenetv3_labels

model_names = ['small', 'large']
parser = argparse.ArgumentParser()
parser.add_argument(
    '-m', '--model', default='small', choices=model_names,
    help='model architecture (default: small)'
)
args = parser.parse_args()

weight_path = 'mobilenetv3_' + args.model + '.onnx'
model_path = weight_path + '.prototxt'

img_path = './clock.jpg'

# model download
rmt_ckpt = "https://storage.googleapis.com/ailia-models/mobilenetv3/"
if not os.path.exists(model_path):
    print("downloading ...")
    urllib.request.urlretrieve(rmt_ckpt + model_path, model_path)
if not os.path.exists(weight_path):
    urllib.request.urlretrieve(rmt_ckpt + weight_path, weight_path)

# classifier initialize
env_id = ailia.get_gpu_environment_id()
net = ailia.Net(model_path, weight_path, env_id=env_id)

# load input image and convert to BGRA
print("inferencing ...")

img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

# preprocessing
mean = [0.485, 0.456, 0.406]  # mean of ImageNet dataset
std = [0.229, 0.224, 0.225]  # std of ImageNet dataset
img = cv2.resize(img, (256, 256))  # resize image
# center clop & normalize between 0 and 1
img = np.array(img[16:240, 16:240], dtype='float64') / 255  
for i in range(3):  # normalize image
    img[:, :, i] = (img[:, :, i] - mean[i]) / std[i]

# [x, y, channel] --> [1, channel, x, y]
img = np.expand_dims(np.rollaxis(img, 2, 0), axis=0) 
# print(img.shape)

# compute
max_class_count = 3
preds_ailia = net.predict(img)[0]

# get result
top = preds_ailia.argsort()[-1 * max_class_count:][::-1]

print("class_count={}".format(max_class_count))

for idx in range(max_class_count) :
    # print result
    print("+ idx=" + str(idx))
    print("  category={}".format(top[idx]) + "[ " +
          mobilenetv3_labels.imagenet_category[top[idx]] + " ]" )
    print("  prob={}".format(preds_ailia[top[idx]]))

