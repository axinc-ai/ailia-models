# MobilenetV2

## Input

![Input](clock.jpg)

Ailia input shape : (1,3,224,224)  

## Output
```
class_count=3
+ idx=0
  category=409[analog clock ]
  prob=20.156919479370117
+ idx=1
  category=892[wall clock ]
  prob=17.56859588623047
+ idx=2
  category=426[barometer ]
  prob=13.731719017028809
```

## usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 mobilenetv2.py
```

If you want to specify the input image, put the image path after the `--input` option.  
```bash
$ python3 mobilenetv2.py --input IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 mobilenetv2.py --video VIDEO_PATH
```

The default setting is to use the optimized model and weights, but you can also switch to the normal model by using the --normal option.

## Reference

[PyTorch Implemention of MobileNet V2](https://github.com/d-li14/mobilenetv2.pytorch)

## Model Format

ONNX opset = 10

## Framework

Pytorch 1.2.0

## Netron

[mobilenetv2_1.0.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/mobilenetv2/mobilenetv2_1.0.onnx.prototxt)

[mobilenetv2_1.0.opt.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/mobilenetv2/mobilenetv2_1.0.opt.onnx.prototxt)
