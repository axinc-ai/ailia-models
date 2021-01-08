# Convolutional Recurrent Neural Network

## Input

![Input](demo.png)

(Image from https://github.com/meijieru/crnn.pytorch/blob/master/data/demo.png)

Shape : (1, 1, 32, 100)  

## Output

A string consisting of an array of the characters in "0123456789abcdefghijklmnopqrstuvwxyz"

Shape : (N, 1, 37)

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 crnn_pytorch.py

a-----v--a-i-l-a-bb-l-e--- => available
```

If you want to specify the input image, put the image path after the `--input` option.
```bash
$ python3 crnn_pytorch.py --input IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 crnn_pytorch.py --video VIDEO_PATH
```

## Reference

- [Convolutional Recurrent Neural Network](https://github.com/meijieru/crnn.pytorch)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[crnn.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/crnn_pytorch/crnn.onnx.prototxt)
