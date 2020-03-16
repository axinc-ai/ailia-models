# yolov3

## Input

![Input](couple.jpg)

Shape : (1, 3, 448, 448)
Range : [0.0, 1.0]

## Output

![Output](output.png)

- category : [0,79]
- probablity : [0.0,1.0]
- position : x, y, w, h [0,1]

## Reference

- [ONNX Model Zoo](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov3)

## Framework

ONNX Runtime

## Model Format

ONNX opset=10

## Netron

[yolov3.opt.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/yolov3/yolov3.opt.onnx.prototxt)
