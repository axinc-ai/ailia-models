# GoogleNet

## Input

![Input](pizza.jpg)

Shape : (1,3,224,224)
Range : [0.0, 1.0]

## Output

```
+ idx=0
  category=963[ pizza, pizza pie ]
  prob=7.194718837738037
+ idx=1
  category=926[ hot pot, hotpot ]
  prob=6.815596103668213
+ idx=2
  category=567[ frying pan, frypan, skillet ]
  prob=6.665373802185059
```

## Reference

[Going Deeper with Convolutions]( https://arxiv.org/abs/1409.4842 )

[GOOGLENET]( https://pytorch.org/hub/pytorch_vision_googlenet/)

## Model Format

ONNX opset = 10

## Framework

Pytorch 1.3.0
