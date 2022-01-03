# AlexNet

## Input
#### Image mode
Input file type must be .jpg image.
#### Video mode
Input file type must be .mp4 video.

![Input](clock.jpg)

## Output
```
class_count=3
+ idx=0
  category=409[analog clock ]
  prob=0.8660851716995239
+ idx=1
  category=892[wall clock ]
  prob=0.13281846046447754
+ idx=2
  category=826[stopwatch, stop watch ]
  prob=0.0008276691660284996
```

## Usage

Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet while downloading.

For the sample image,

```
$ python3 alexnet.py --input [your sample.jpg path]
```

For the sample video,

```
$ python3 alexnet.py --video [your sample.mp4 path]
```

## Reference

[AlexNet | PyTorch](https://pytorch.org/hub/pytorch_vision_alexnet/)

## Model Format

ONNX opset = 11

## Framework

Pytorch 1.7.1

## Netron

[alexnet.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/alexnet/alexnet.onnx.prototxt)
