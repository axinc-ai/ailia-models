# yolov2

## Input

![Input](couple.jpg)

Shape : (1, 3, 448, 448)
Range : [-0.5, 0.5]

## Output

![Output](output.png)

- category : [0,79]
- probablity : [0.0,1.0]
- position : x, y, w, h [0,1]

## Reference

- [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolov2/)
- [Covert original YOLO model from Pytorch to Onnx, and do inference using backend Caffe2 or Tensorflow.](https://github.com/purelyvivid/yolo2_onnx)

## Framework

Pytorch 1.3.1

## Model Format

ONNX opset=10
