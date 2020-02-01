import os, urllib, time

import cv2
import numpy as np

import ailia


# FIXME [WARNING] crop (512, 512) is better than resize automatically
img_path = 'test.jpg'
width, height = 512, 512  # fixed when exporting model

# TODO add squeezenet version if necessary
model_path = 'pspnet-hair-segmentation.onnx'
weight_path = model_path + '.prototxt'

# downloading
rmt_ckpt =\
    'https://storage.googleapis.com/ailia-models/pspnet-hair-segmentation/'
if not os.path.exists(model_path):
    print('model downloading...')
    urllib.request.urlretrieve(rmt_ckpt + model_path, model_path)
if not os.path.exists(weight_path):
    urllib.request.urlretrieve(rmt_ckpt + weight_path, weight_path)


def main():
    # create model
    env_id = ailia.get_gpu_environment_id()
    print("Environment mode: {} (-1: CPU, 1: GPU)".format(env_id))
    net = ailia.Net(model_path, weight_path, env_id=env_id)

    # image processing
    org_img = cv2.imread(img_path) / 255.0
    org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

    # image preprocessing
    mean = [0.485, 0.456, 0.406]  # mean of ImageNet dataset
    std = [0.229, 0.224, 0.225]  # std of ImageNet dataset
    img = cv2.resize(org_img, (height, width))  # resize image

    for i in range(3):  # normalize image
        img[:, :, i] = (img[:, :, i] - mean[i]) / std[i]

    # [x, y, channel] --> [1, channel, x, y]
    img = np.expand_dims(np.rollaxis(img, 2, 0), axis=0) 
    
    # net.set_input_shape(img.shape)  # TODO remove

    # compute time
    for i in range(10):
        start = int(round(time.time() * 1000))
        pred = net.predict(img)
        end = int(round(time.time() * 1000))
        print(f"ailia processing time {end-start} ms")

    # prepare mask
    print(f'[DEBUG] {pred.shape}')
    
    # output saving
    print('Successfully finished!')


if __name__ == "__main__":
    main()
