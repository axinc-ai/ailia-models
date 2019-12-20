# MobilenetV2

## Input

![Input](clock.jpg)

Shape : (256,256)  -> (1,3,224,224)  (after preprocessing)
Range : [0.0, 1.0]

## Output
```
inferencing ...
class_count=3
+ idx=0
  category=409[ analog clock ]
  prob=23.100770950317383
+ idx=1
  category=892[ wall clock ]
  prob=20.599042892456055
+ idx=2
  category=426[ barometer ]
  prob=17.74355125427246
```

## Reference

[PyTorch Implemention of MobileNet V2](https://github.com/d-li14/mobilenetv2.pytorch)

## Model Format

ONNX opset = 10

## Framework

Pytorch 1.2.0
