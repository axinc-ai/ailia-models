# Resnet18

## Input

![Input](pizza.jpg)

Ailia input shape : (1,3,224,224)  
Range : [-127.0, 127.0]

## Output

```
+ idx=0
  category=963[pizza, pizza pie ]
  prob=0.573032021522522
+ idx=1
  category=927[trifle ]
  prob=0.3181869089603424
+ idx=2
  category=415[bakery, bakeshop, bakehouse ]
  prob=0.02643408812582493
```

### usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 resnet18.py
```

If you want to specify the input image, put the image path after the `--input` option.  
```bash
$ python3 resnet18.py --input IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 resnet18.py --video VIDEO_PATH
```

## Reference

[ResNet18](https://pytorch.org/vision/main/generated/torchvision.models.resnet18.html)

## Model Format

ONNX opset = 11

## Framework

pytorch 1.10

## Netron

[resnet18.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/resnet18/resnet18.onnx.prototxt)
