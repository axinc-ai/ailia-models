# EfficientNet

## Input

![Input](clock.jpg)

Ailia input shape : (1,3,224,224)  
Range : [-1.0, 1.0]

## Output

```
+ idx=0
  category=409 [ analog clock ]
  prob=9.720746994018555
+ idx=1
  category=892 [ wall clock ]
  prob=6.404201030731201
+ idx=2
  category=426 [ barometer ]
  prob=4.357946395874023
```

### usage

Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet while downloading.

For the sample image,
```
$ python3 efficientnet.py
```
If you want to specify the model, put b0 or b7 after the `--model` option.
you can select efficientnet-b0 or efficientnet-b7.
efficientnet-b0 is faster than efficientnet-b7 but lower precision.
```
$ python3 efficientnet.py --model b0
or
$ python3 efficientnet.py --model b7
```

If you want to specify the input image, put the image path after the `--input` option.
```
$ python3 efficientnet.py --input IMAGE_PATH
```
By adding the `--video` option, you can input the video.
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```
$ python3 efficientnet.py --video VIDEO_PATH
```

## Reference

[A PyTorch implementation of EfficientNet]( https://github.com/lukemelas/EfficientNet-PyTorch)

## Model Format

ONNX opset = 10

## Framework

Pytorch 1.1.0

## Netron

[efficientnet-b0.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/efficientnet/efficientnet-b0.onnx.prototxt)

[efficientnet-b7.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/efficientnet/efficientnet-b7.onnx.prototxt)