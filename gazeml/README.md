# GazeML

## Input

![Input](eye.png)

Shape : (2, 36, 60, 1)
Range : [-1.0, 1.0]

## Output

![Output](output.png)

Shape : (2, 36, 60, 18)
Range : [0, 1.0]

## Reference

[A deep learning framework based on Tensorflow for the training of high performance gaze estimation](https://github.com/swook/GazeML)

## Framework

TensorFlow 1.13.1

## Model Format

ONNX opset = 10

## Netron

[gazeml_elg_i60x36_n32.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/gazeml/gazeml_elg_i60x36_n32.onnx.prototxt)

[gazeml_elg_i180x108_n64.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/gazeml/gazeml_elg_i180x108_n64.onnx.prototxt)
