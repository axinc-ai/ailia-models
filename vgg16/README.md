# VGG16

## Input

![Input](pizza.jpg)

Shape : (1,3,224,224)
Range : [-128, 127]

## Output

```
+ idx=0
  category=963[ pizza, pizza pie ]
  prob=0.9207707047462463
+ idx=1
  category=567[ frying pan, frypan, skillet ]
  prob=0.03415448218584061
+ idx=2
  category=964[ potpie ]
  prob=0.024192284792661667
```

## Reference

[Very Deep Convolutional Networks for Large-Scale Image Recognition]( https://arxiv.org/abs/1409.1556 )

[Keras Applications : VGG16]( https://keras.io/applications/#vgg16 )

[keras2caffe]( https://github.com/uhfband/keras2caffe)

## Model Format

CaffeModel

## Framework

Keras
