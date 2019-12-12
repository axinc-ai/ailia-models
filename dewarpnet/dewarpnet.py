import cv2
import time
import urllib.request
import os
import numpy as np

import ailia


img_path = "test.png"

bm_weight_path = "bm_model.onnx"
wc_weight_path = "wc_model.onnx"
bm_model_path = bm_weight_path + ".prototxt"
wc_model_path = wc_weight_path + ".prototxt"


rmt_ckpt = "https://storage.googleapis.com/ailia-models/dewarpnet/"

print("loading model...")

# BM
if not os.path.exists(bm_model_path):
    urllib.request.urlretrieve(rmt_ckpt + bm_model_path, bm_model_path)
if not os.path.exists(bm_weight_path):
    urllib.request.urlretrieve(rmt_ckpt + bm_weight_path, bm_weight_path)

# WC
if not os.path.exists(wc_model_path):
    urllib.request.urlretrieve(rmt_ckpt + wc_model_path, wc_model_path)
if not os.path.exists(wc_weight_path):
    urllib.request.urlretrieve(rmt_ckpt + wc_weight_path, wc_weight_path)


def grid_sample(img, grid):
    height, width, c = img.shape
    output = np.zeros_like(img)
    grid[:, :, 0] = (grid[:, :, 0] + 1) * (width-1) / 2
    grid[:, :, 1] = (grid[:, :, 1] + 1) * (height-1) / 2
    # TODO speed up here
    for h in range(height):
        for w in range(width):
            h_ = int(grid[h, w, 1])
            w_ = int(grid[h, w, 0])
            output[h, w] = img[h_, w_]
    return output


def unwarp(img, bm):
    w,h=img.shape[0],img.shape[1]
    bm = bm.transpose(1, 2, 0)
    bm0=cv2.blur(bm[:,:,0],(3,3))
    bm1=cv2.blur(bm[:,:,1],(3,3))
    bm0=cv2.resize(bm0,(h,w))
    bm1=cv2.resize(bm1,(h,w))
    bm=np.stack([bm0,bm1],axis=-1)
    img = img.astype(float) / 255.0
    res = grid_sample(img, bm)
    return res


wc_img_size = (256, 256)
bm_img_size = (128, 128)

img_org = cv2.imread(img_path)
img = cv2.resize(img_org, wc_img_size)
img = img[:, :, ::-1] / 255.0
img = img.transpose(2, 0, 1)
img = np.expand_dims(img, 0)


# net initialize
env_id = ailia.get_gpu_environment_id()
print("Environment mode: {} (-1: CPU, 1: GPU)".format(env_id))

bm_net = ailia.Net(bm_model_path, bm_weight_path, env_id=env_id)
wc_net = ailia.Net(wc_model_path, wc_weight_path, env_id=env_id)

# compute time
for i in range(1):
    start = int(round(time.time() * 1000))
    
    wc_output = wc_net.predict(img)[0]
    pred_wc = np.clip(wc_output, 0, 1.0).transpose(1, 2, 0)
    bm_input = cv2.resize(pred_wc, bm_img_size).transpose(2, 0, 1)
    bm_input = np.expand_dims(bm_input, 0)
    outputs_bm = bm_net.predict(bm_input)[0]
    uwpred = unwarp(img_org, outputs_bm)  # This is not on GPU!
    
    end = int(round(time.time() * 1000))
    print("ailia processing time {} ms".format(end-start))

cv2.imwrite('output.png', uwpred * 255)
    
print('Successfully finished !')
