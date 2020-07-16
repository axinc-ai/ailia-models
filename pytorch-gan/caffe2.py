#!/bin/python3

import onnx
import caffe2.python.onnx.backend as onnx_caffe2_backend
import numpy
import onnxruntime as rt

# Load the ONNX ModelProto object. model is a standard Python protobuf object
modelpath = "./netg_exported_mod.onnx"
model = onnx.load(modelpath)

# check the model
onnx.checker.check_model(model)

prepared_backend = onnx_caffe2_backend.prepare(model)

# run the model in Caffe2

# Construct a map from input names to Tensor data.
# The graph of the model itself contains inputs for all weight parameters, after the input image.
# Since the weights are already embedded, we just need to pass the input image.
# Set the first input.
x = numpy.random.rand(1,512).astype(numpy.float32)
W = {model.graph.input[0].name: x}

# Run the Caffe2 net:
c2_out = prepared_backend.run(W)[0]

outp = numpy.clip((0.5 + 255*c2_out.transpose((2,3,1,0)).reshape((64,64,3))).astype(numpy.float32),0,255)

from PIL import Image
img = Image.fromarray(outp.astype(numpy.uint8),'RGB')
img.show()
