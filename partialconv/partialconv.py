import time
import cv2
import os
import urllib.request
import numpy as np
import argparse

import ailia


model_names = ['resnet50', 'vgg16_bn', 'pdresnet50', 'pdresnet101', 'pdresnet152']
img_name = "test_5735.JPEG"

parser = argparse.ArgumentParser()
parser.add_argument('--arch', '-a', metavar='ARCH', default='pdresnet152', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: pdresnet152)')
args = parser.parse_args()
model_name = args.arch

# label of 1000 classes
LABEL = {}
with open("label.txt") as f:
    for line in f:
        (key, val) = line.split(':')
        LABEL[int(key)] = val

weight_path = model_name + ".onnx"
model_path = weight_path + ".prototxt"

if not os.path.exists(model_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/ailia-models/partialconv/" + model_path, model_path)
if not os.path.exists(weight_path):
    urllib.request.urlretrieve("https://storage.googleapis.com/ailia-models/partialconv/" + weight_path, weight_path)

print(weight_path)

mean = [0.485, 0.456, 0.406]  # mean of ImageNet dataset
std = [0.229, 0.224, 0.225]  # std of ImageNet dataset

img = cv2.imread(img_name)

"""
======================================================================
 Here is a special image preprocessing for imagenet dataset 
======================================================================
 """
img = cv2.resize(img, (256, 256))  # resize image
img = np.array(img[16:240, 16:240], dtype='float64') / 255  # center clop & normalize between 0 and 1
for i in range(3):  # normalize image
    img[:, :, i] = (img[:, :, i] - mean[i]) / std[i]
"""
======================================================================
 Until Here
======================================================================
"""

img = np.expand_dims(np.rollaxis(img, 2, 0), axis=0)  # [x, y, channel] --> [1, channel, x, y]

# net initialize
env_id = ailia.get_gpu_environment_id()
net = ailia.Net(model_path, weight_path, env_id=env_id)
# net.set_input_shape((1, 3, 224, 224))
print(net.get_summary())

# compute time
for i in range(10):
    start = int(round(time.time() * 1000))
    preds_ailia = net.predict(img)[0]
    end = int(round(time.time() * 1000))
    print("ailia processing time {} ms".format(end-start))

print("The predicted label is " + LABEL[preds_ailia.argmax()])
