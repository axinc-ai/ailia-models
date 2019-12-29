import cv2
import time
import os
import urllib.request
from matplotlib import pyplot as plt
import numpy as np

import ailia

# TODO adapting 96 and 226

# img_path = input("input image name: ")
img_path = "test.png"
weight_path = "resnet_facial_feature.onnx"
model_path = "resnet_facial_feature.onnx.prototxt"

rmt_ckpt = "https://storage.googleapis.com/ailia-models/resnet_facial_feature/"

if not os.path.exists(model_path):
    urllib.request.urlretrieve(rmt_ckpt + model_path, model_path)
if not os.path.exists(weight_path):
    urllib.request.urlretrieve(rmt_ckpt + weight_path, weight_path)


# load dataset
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (226, 226)).reshape(1, 1, 226, 226) / 255

# net initialize
env_id = ailia.get_gpu_environment_id()
print(env_id)
net = ailia.Net(model_path, weight_path, env_id=env_id)

# compute time
for i in range(10):
    start = int(round(time.time() * 1000))
    preds_ailia = net.predict(img)[0]
    end = int(round(time.time() * 1000))
    print("ailia processing time {} ms".format(end-start))

fig = plt.figure(figsize=(3, 3))
plt.show()
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(img.reshape(226, 226))
points = np.vstack(np.split(preds_ailia, 15)).T * 113 + 113
ax.plot(points[0], points[1], 'o', color='red')
fig.savefig('output.png')
