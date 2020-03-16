# 2D and 3D Face alignment library build using pytorch

## Input

![Input](aflw-test.jpg)

(from https://github.com/1adrianb/face-alignment/tree/master/test/assets)

Shape : (1, 3, 256, 256)
Range : [0.0, 1.0]

## Output

![Output](output.png)

Confidence : (1, 68, 64, 64)

## Reference

[2D and 3D Face alignment library build using pytorch](https://github.com/1adrianb/face-alignment)

## Framework

Pytorch 1.2.0

## Model Format

ONNX opset = 10

## Netron

[face_alignment.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/face_alignment/face_alignment.onnx.prototxt)
