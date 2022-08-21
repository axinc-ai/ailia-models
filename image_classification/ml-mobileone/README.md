# MobileOne

## Input

![Input](./input.jpg)

Ailia input shape : (1, 3, 256, 256)

## Output

Output is as below.
```
==============================================================
class_count=3
+ idx=0
  category=981[ballplayer, baseball player ]
  prob=9.265481948852539
+ idx=1
  category=429[baseball ]
  prob=5.025822639465332
+ idx=2
  category=615[knee pad ]
  prob=4.340822696685791
```

## Usage
Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet while downloading.

For the sample image,

```
$ python3 ml_mobileone.py
```

If you want to specify the input image, put the image path after the --input option.

```
$ python3 ml_mobileone.py --input IMAGE_PATH
```

By adding the --video option, you can input the video.
If you pass 0 as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.

```
$ python3 ml_mobileone.py --video VIDEO_PATH
```

You can select a model from `s0 | s1 | s2 | s3 | s4` by adding --model option.

## Reference

[A PyTorch implementation of MobileOne](https://github.com/apple/ml-mobileone)

[IMAGENET](https://image-net.org/)

[ImageNet 1000 (mini)](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000)

## Model Format

ONNX opset = 11

## Framework

Pytorch 1.11.0

## Netron

[mobileone_s0.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/ml-mobileone/mobileone_s0.onnx.prototxt)

[mobileone_s1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/ml-mobileone/mobileone_s1.onnx.prototxt)

[mobileone_s2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/ml-mobileone/mobileone_s2.onnx.prototxt)

[mobileone_s3.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/ml-mobileone/mobileone_s3.onnx.prototxt)

[mobileone_s4.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/ml-mobileone/mobileone_s4.onnx.prototxt)