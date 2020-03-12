import time
import os
import urllib.request

import cv2
import sys
import numpy as np

import ailia


# TODO Video mode
IMG_PATH = 'input.jpg'

MODEL_NAME = 'monodepth2_mono+stereo_640x192'

ENC_WEIGHT_PATH = MODEL_NAME + '_enc.onnx'
ENC_MODEL_PATH = MODEL_NAME + '_enc.onnx.prototxt'
DEC_WEIGHT_PATH = MODEL_NAME + '_dec.onnx'
DEC_MODEL_PATH = MODEL_NAME + '_dec.onnx.prototxt'

RMT_CKPT = "https://storage.googleapis.com/ailia-models/monodepth2/"


if not os.path.exists(ENC_MODEL_PATH):
    print('enocder model downloading...')
    urllib.request.urlretrieve(RMT_CKPT + ENC_MODEL_PATH, ENC_MODEL_PATH)
if not os.path.exists(ENC_WEIGHT_PATH):
    urllib.request.urlretrieve(RMT_CKPT + ENC_WEIGHT_PATH, ENC_WEIGHT_PATH)

if not os.path.exists(DEC_MODEL_PATH):
    print('decoder model downloading...')
    urllib.request.urlretrieve(RMT_CKPT + DEC_MODEL_PATH, DEC_MODEL_PATH)
if not os.path.exists(DEC_WEIGHT_PATH):
    urllib.request.urlretrieve(RMT_CKPT + DEC_WEIGHT_PATH, DEC_WEIGHT_PATH)


def main():
    pass


if __name__ == "__main__":
    main()
