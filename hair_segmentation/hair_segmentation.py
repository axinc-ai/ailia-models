import cv2
import time
import urllib.request
import os
import numpy as np

import ailia


def transfer(image, mask):
    mask[mask > 0.5] = 255
    mask[mask <= 0.5] = 0

    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    mask_n = np.zeros_like(image)
    mask_n[:, :, 0] = mask

    alpha = 0.8
    beta = (1.0 - alpha)
    dst = cv2.addWeighted(image, alpha, mask_n, beta, 0.0)

    return dst


# img_path = input("image name (00041.jpg, 00053.jpg): ")
img_path = "input.jpg"

weight_path = "hair_segmentation.onnx"
model_path = weight_path + ".prototxt"

rmt_ckpt = "https://storage.googleapis.com/ailia-models/hair_segmentation/"

print("loading model...")


if not os.path.exists(model_path):
    urllib.request.urlretrieve(rmt_ckpt + model_path, model_path)
if not os.path.exists(weight_path):
    urllib.request.urlretrieve(rmt_ckpt + weight_path, weight_path)

# preprocessing
img = cv2.imread(img_path)
input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_img = cv2.resize(input_img, (224, 224))
input_img = input_img / 255
input_img = input_img.reshape((1,) + input_img.shape)
# print(input_img.shape) -> (1, 224, 224, 3)


# net initialize
env_id = ailia.get_gpu_environment_id()
print("Environment mode: {} (-1: CPU, 1: GPU)".format(env_id))

net = ailia.Net(model_path, weight_path, env_id=env_id)
net.set_input_shape(input_img.shape)

# compute time
for i in range(10):
    start = int(round(time.time() * 1000))
    pred = net.predict(input_img)
    end = int(round(time.time() * 1000))
    print("ailia processing time {} ms".format(end-start))

pred = pred.reshape((224, 224))
dst = transfer(img, pred)
cv2.imwrite("output.png", dst)

print('Successfully finished !')
