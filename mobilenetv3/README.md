# MobilenetV3

## Input

![Input](clock.jpg)

Ailia input shape : (1,3,224,224)

## Output
```bash
class_count=3
+ idx=0
  category=409[analog clock ]
  prob=16.309402465820312
+ idx=1
  category=892[wall clock ]
  prob=10.951416015625
+ idx=2
  category=816[spindle ]
  prob=8.520172119140625
```

## usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 mobilenetv3.py
```

If you want to specify the input image, put the image path after the `--input` option.  
```bash
$ python3 mobilenetv3.py --input IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 mobilenetv3.py --video VIDEO_PATH
```

You can select a pretrained model by specifying `-a large` or `-a small`(default).


## Reference

[PyTorch Implemention of MobileNet V3](https://github.com/d-li14/mobilenetv3.pytorch)

## Model Format

ONNX opset = 10

## Framework

Pytorch

## Netron

[mobilenetv3_small.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/mobilenetv3/mobilenetv3_small.onnx.prototxt)
