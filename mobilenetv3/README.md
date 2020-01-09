# MobilenetV3

## Input

![Input](clock.jpg)

Shape : (256,256)  -> (1,3,224,224)  (after preprocessing)
Range : [0.0, 1.0]

## Output
```bash
inferencing ...
class_count=3
+ idx=0
  category=409[ analog clock ]
  prob=15.144688606262207
+ idx=1
  category=892[ wall clock ]
  prob=11.657188415527344
+ idx=2
  category=816[ spindle ]
  prob=8.147628784179688
```

## Reference

[PyTorch Implemention of MobileNet V3](https://github.com/d-li14/mobilenetv3.pytorch)

## Model Format

ONNX opset = 10

## Framework

Pytorch
