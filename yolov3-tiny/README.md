# yolov1-tiny

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

- [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/)
- [A Keras implementation of YOLOv3 (Tensorflow backend)](https://github.com/qqwweee/keras-yolo3)
- [keras-onnx](https://github.com/onnx/keras-onnx/tree/master/applications/yolov3)

## Framework

Keras 2.2.4

## Model Format

ONNX opset=10
