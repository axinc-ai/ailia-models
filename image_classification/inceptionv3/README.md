# InceptionV3

## Input

![Input](clock.jpg)

Ailia input shape: (1,3,299,299)
Range: [0.0, 255.0]

## Output
```
class_count=3
+ idx=0
category=409[ analog clock ]
prob=9.799751281738281
+ idx=1
category=892[ wall clock ]
prob=7.499673843383789
+ idx=2
category=826[ stopwatch, stop watch ]
prob=4.118775844573975
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 inceptionv3.py
```

If you want to specify the input image, put the image path after the `--input` option.  
```bash
$ python3 inceptionv3.py --input IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 inceptionv3.py --video VIDEO_PATH
```

## Reference

[Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567)

[Datasets, Transforms and Models specific to Computer Vision](https://github.com/pytorch/vision)

## Model Format

ONNX opset = 10

## Framework

Pytorch 1.2.0

## Netron

[inceptionv3.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/inceptionv3/inceptionv3.onnx.prototxt)
