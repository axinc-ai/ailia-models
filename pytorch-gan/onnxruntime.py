#!/bin/python3

import numpy
import os
import cv2
from PIL import Image

import onnxruntime as rt
import onnx

modelpath = "./netg_exported.onnx"

onnx_sess = rt.InferenceSession(modelpath)

onnx_model = onnx.load(modelpath)

input_name = onnx_sess.get_inputs()[0].name
onnx_pred = 255 * onnx_sess.run(None, {input_name: numpy.random.rand(1,512).astype(numpy.float32)})[0].astype(numpy.float32)
# (without the astype(.) it is float64 and incompatible with onnxruntime)

img = Image.fromarray(onnx_pred.transpose((2,3,1,0)).reshape((64,64,3)).astype(numpy.uint8),'RGB')
img.show()
