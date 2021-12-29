# AlexNet

## Input
#### Image mode
Input file type must be .jpg image.
#### Video mode
Input file type must be .mp4 video.

![Input](input/dog.jpg)

## Output
```
[Image_1] input/dog.jpg
	Samoyed 0.7245363593101501
	wallaby 0.13860034942626953
	Pomeranian 0.05935253947973251
	Angora 0.022971760481595993
	Arctic fox 0.012396696954965591
```

## Usage

Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet while downloading.

For the sample image,

```
$ python3 alexnet.py --input input/[your sample.jpg path]
```

For the sample video,

```
$ python3 alexnet.py --video input/[your sample.mp4 path]
```

## Reference

[AlexNet | PyTorch](https://pytorch.org/hub/pytorch_vision_alexnet/)

## Model Format

ONNX opset = 11

## Framework

Pytorch 1.7.1

## Netron

[alexnet.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/alexnet/alexnet.onnx.prototxt)
