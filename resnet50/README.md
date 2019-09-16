# Resnet50

## Input

![Input](pizza.jpg)

Shape : (1,3,224,224)
Range : [-127.0, 127.0]

## Output

```
+ idx=0
  category=963[ pizza, pizza pie ]
  prob=0.8783312439918518
+ idx=1
  category=927[ trifle ]
  prob=0.04941209405660629
+ idx=2
  category=567[ frying pan, frypan, skillet ]
  prob=0.011235987767577171
```

## Reference

[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://github.com/twtygqyy/pytorch-SRResNet)

## Model Format

ONNX opset = 10

## Framework

Pytorch 1.2.0
