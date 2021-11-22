import cv2
import numpy as np

def preprocess_input(im, height, width):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    im = cv2.resize(im, dsize=(height, width)).astype(np.float32) / 255

    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    im = (im - mean) / std

    im = im.transpose(2, 0, 1)
    im = np.expand_dims(im, 0)
    return im

def softmax(x):
    f = np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims = True)

    return f