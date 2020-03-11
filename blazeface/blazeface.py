import time
import os
import urllib.request

import cv2
import sys
import numpy as np

import ailia
from utils import *

MODE = "image"
if len(sys.argv) >= 2:
    MODE = sys.argv[1]
    if MODE != "image" and MODE != "video":
        print("please set mdoe to image or video")
        sys.exit()

img_path = 'input.png'

weight_path = 'blazeface.onnx'
model_path = 'blazeface.onnx.prototxt'

rmt_ckpt = "https://storage.googleapis.com/ailia-models/blazeface/"

if not os.path.exists(model_path):
    urllib.request.urlretrieve(rmt_ckpt + model_path, model_path)
if not os.path.exists(weight_path):
    urllib.request.urlretrieve(rmt_ckpt + weight_path, weight_path)

# net initialize
env_id = ailia.get_gpu_environment_id()
print(f'env_id (0: cpu, 1: gpu): {env_id}')
net = ailia.Net(model_path, weight_path, env_id=env_id)


def recognize_from_image():
    # prepare input data
    org_img = cv2.imread(img_path)
    org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    imgs = load_image(img_path)

    # compute time
    for i in range(5):
        start = int(round(time.time() * 1000))
        # preds_ailia = net.predict(imgs)
        input_blobs = net.get_input_blob_list()
        net.set_input_blob_data(imgs, input_blobs[0])
        net.update()
        preds_ailia = net.get_results()
        end = int(round(time.time() * 1000))
        print("ailia processing time {} ms".format(end - start))

    # Postprocess
    detections = postprocess(preds_ailia)

    # generate detections
    for detection in detections:
        plot_detections(org_img, detection)

    # show_result(org_img,detections)
    # cv2.imwrite( "result.png", org_img)


def letterbox_convert(img):
    tmp = img[:, :]
    height, width = img.shape[:2]
    if(height > width):
        size = height
        limit = width
    else:
        size = width
        limit = height
    start = int((size - limit) / 2)
    fin = int((size + limit) / 2)
    img = cv2.resize(np.zeros((1, 1, 3), np.uint8), (size, size))
    if(size == height):
        img[:, start:fin] = tmp
    else:
        img[start:fin, :] = tmp
    return img


def show_result(input_img, detections):
    for detection in detections:
        for d in detection:
            w = input_img.shape[1]
            h = input_img.shape[0]
            top_left = (int(d[1]*w), int(d[0]*h))
            bottom_right = (int(d[3]*w), int(d[2]*h))
            color = (255, 255, 255)
            cv2.rectangle(input_img, top_left, bottom_right, color, 4)

            for k in range(6):
                kp_x = d[4 + k*2] * input_img.shape[1]
                kp_y = d[4 + k*2 + 1] * input_img.shape[0]
                r = int(input_img.shape[1]/100)
                cv2.circle(input_img, (int(kp_x), int(kp_y)),
                           r, (255, 255, 255), -1)


def recognize_from_video():
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("webcamera not found")
        sys.exit()
    while(True):
        ret, frame = capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if not ret:
            continue

        input_img = letterbox_convert(frame)

        img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))

        imgs = np.expand_dims(np.rollaxis(img, 2, 0),
                              axis=0).astype(np.float32)
        imgs = imgs / 127.5 - 1.0

        input_blobs = net.get_input_blob_list()
        net.set_input_blob_data(imgs, input_blobs[0])
        net.update()
        preds_ailia = net.get_results()

        detections = postprocess(preds_ailia)
        show_result(input_img, detections)

        cv2.imshow('frame', input_img)

    capture.release()
    cv2.destroyAllWindows()


if MODE == "image":
    recognize_from_image()

if MODE == "video":
    recognize_from_video()
