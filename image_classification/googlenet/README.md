# GoogleNet

## Input

![Input](pizza.jpg)

Ailia input shape: (224, 224, 4)  
Range : [0.0, 255.0]  (np.uint8)

## Output

```
+ idx=0
  category=963[ pizza, pizza pie ]
  prob=7.194718837738037
+ idx=1
  category=926[ hot pot, hotpot ]
  prob=6.815596103668213
+ idx=2
  category=567[ frying pan, frypan, skillet ]
  prob=6.665373802185059
```

## Usage
Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet while downloading.

For the sample image,
```
$ python3 googlenet.py
```

If you want to specify the input image, put the image path after the `--input` option.
```
$ python3 googlenet.py --input IMAGE_PATH
```
By adding the `--video` option, you can input the video.
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```
$ python3 googlenet.py --video VIDEO_PATH
```


## Reference

[Going Deeper with Convolutions]( https://arxiv.org/abs/1409.4842 )

[GOOGLENET]( https://pytorch.org/hub/pytorch_vision_googlenet/)

## Model Format

ONNX opset = 10

## Framework

Pytorch 1.3.0

## Netron

[googlenet.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/googlenet/googlenet.onnx.prototxt)
