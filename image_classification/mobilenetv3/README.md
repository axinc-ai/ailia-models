# MobilenetV3

## Input

![Input](clock.jpg)

Ailia input shape : (1,3,224,224)

## Output
- large model
```bash
class_count=3
+ idx=0
  category=409[analog clock ]
  prob=10.01405143737793
+ idx=1
  category=892[wall clock ]
  prob=7.830683708190918
+ idx=2
  category=530[digital clock ]
  prob=3.3934028148651123
```

- small model
```bash
class_count=3
+ idx=0
  category=409[analog clock ]
  prob=19.85206413269043
+ idx=1
  category=892[wall clock ]
  prob=19.0461368560791
+ idx=2
  category=826[stopwatch, stop watch ]
  prob=12.139641761779785
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

[mobilenetv3_small.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/mobilenetv3/mobilenetv3_small.onnx.prototxt)
[mobilenetv3_large.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/mobilenetv3/mobilenetv3_large.onnx.prototxt)
