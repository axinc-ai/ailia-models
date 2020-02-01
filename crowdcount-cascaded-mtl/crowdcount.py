import time
import os
import urllib.request

import cv2
import numpy as np

import ailia

# TODO adapting 96 and 226

# img_path = input("input image name: ")
img_path = "test.jpeg"
weight_path = "crowdcount.onnx"
model_path = "crowdcount.onnx.prototxt"

rmt_ckpt = "https://storage.googleapis.com/ailia-models/crowd_count/"

if not os.path.exists(model_path):
    urllib.request.urlretrieve(rmt_ckpt + model_path, model_path)
if not os.path.exists(weight_path):
    urllib.request.urlretrieve(rmt_ckpt + weight_path, weight_path)


def read_image(fname):
    img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32, copy=False)
    # Img size is decided when exporting model (modifiable when exporting)
    img = cv2.resize(img, (640, 480))  
    img = img.reshape((1, 1, img.shape[0], img.shape[1]))
    return img


# load image
img = read_image(img_path)

# net initialize
env_id = ailia.get_gpu_environment_id()
print(env_id)
net = ailia.Net(model_path, weight_path, env_id=env_id)

# compute time
for i in range(3):
    start = int(round(time.time() * 1000))
    preds_ailia = net.predict(img)
    end = int(round(time.time() * 1000))
    print("ailia processing time {} ms".format(end - start))

density_map = (255 * preds_ailia / np.max(preds_ailia))[0][0]
res_img = img[0][0]
if density_map.shape != res_img.shape:
    density_map = cv2.resize(density_map, (res_img.shape[1], res_img.shape[0]))
res_img = np.hstack((res_img, density_map))
cv2.imwrite('result.png', res_img)

print('Successfully finished !')
