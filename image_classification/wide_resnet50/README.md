# Wide Resnet50

## Input

![Input](dog.jpg)

(Image from https://github.com/pytorch/hub/raw/master/images/dog.jpg)

Shape : (1,3,224,224)  

## Output

```
+ idx=0
  category=258[Samoyed ]
  prob=0.9123759269714355
+ idx=1
  category=259[Pomeranian ]
  prob=0.030867546796798706
+ idx=2
  category=261[keeshond ]
  prob=0.024926669895648956
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 wide_resnet50.py
```

If you want to specify the input image, put the image path after the `--input` option.  
```bash
$ python3 wide_resnet50 --input IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 wide_resnet50.py --video VIDEO_PATH
```

## Reference

[PYTORCH HUB
FOR RESEARCHERS - WIDE RESNET](https://pytorch.org/hub/pytorch_vision_wide_resnet/)

## Framework

Pytorch

## Model Format

ONNX opset = 11

## Netron

[wide_resnet50_2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/wide_resnet50/wide_resnet50_2.onnx.prototxt)
