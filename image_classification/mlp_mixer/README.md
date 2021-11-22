# MLP-Mixer

## Input

![Input](input.jpg)

Ailia input shape: (1,3,224,224)
Range: [0.0, 255.0]

## Output
```
class_count=3
+ idx=0
  category=5[dog ]
  prob=2.362039566040039
+ idx=1
  category=3[cat ]
  prob=1.0062607526779175
+ idx=2
  category=2[bird ]
  prob=0.6622800827026367
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 mlp_mixer.py
```

If you want to specify the input image, put the image path after the `--input` option.  
```bash
$ python3 mlp_mixer.py --input IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 mlp_mixer.py --video VIDEO_PATH
```

## Reference

[MLP-Mixer](https://github.com/jeonsworld/MLP-Mixer-Pytorch)

## Model Format

ONNX opset = 11

## Framework

pytorch

## Netron

[Mixer-B_16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/mlp_mixer/Mixer-B_16.onnx.prototxt)
[Mixer-L_16.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/mlp_mixer/Mixer-L_16.onnx.prototxt)