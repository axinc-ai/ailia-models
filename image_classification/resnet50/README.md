# Resnet50

## Input

![Input](pizza.jpg)

Ailia input shape : (1,3,224,224)  
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

### usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 resnet50.py
```

If you want to specify the input image, put the image path after the `--input` option.  
```bash
$ python3 ??? --input IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 resnet50.py --video VIDEO_PATH
```

You can select a model from `resnet50.opt | resnet50 | resnet50_pytorch` by adding --arch (default: resnet50.opt).

## Reference

[Deep Residual Learning for Image Recognition]( https://github.com/KaimingHe/deep-residual-networks)

## Model Format

ONNX opset = 10

## Framework

Chainer 6.3.0, Pytorch

## Netron

[resnet50.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/resnet50/resnet50.onnx.prototxt)

[resnet50.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/resnet50/resnet50.opt.onnx.prototxt)

[resnet50_pytorch.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/resnet50/resnet50_pytorch.onnx.prototxt)
