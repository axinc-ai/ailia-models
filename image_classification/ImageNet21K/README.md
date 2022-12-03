# InceptionV4

## Input

![Input](input.jpg)

Ailia input shape: (1,3,299,299)
Range: [0.0, 255.0]

## Output
```
class_count=4
+ idx=0
  category=717[instrument ]
  prob=59.333587646484375
+ idx=1
  category=509[device ]
  prob=44.046539306640625
+ idx=2
  category=160[clock ]
  prob=42.2644157409668
+ idx=3
  category=6[artifact ]
  prob=39.8267822265625
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 imagenet21k.py
```

If you want to specify the input image, put the image path after the `--input` option.  
```bash
$ python3 imagenet21k.py --input IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 imagenet21k.py --video VIDEO_PATH
```

## Reference

[ImageNet21K](https://github.com/Alibaba-MIIL/ImageNet21K)

## Model Format

ONNX opset = 14

## Framework

pytorch

## Netron

[imagenet21k.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/imagenet21k/imagenet21k.onnx.prototxt)
