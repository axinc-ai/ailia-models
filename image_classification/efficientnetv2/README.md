# efficientnetv2

## Input

![Input](input.jpg)

Shape : (1,224,224,3)

## Output

```
class_count=3
+ idx=0
  category=409[analog clock ]
  prob=9.556556701660156
+ idx=1
  category=892[wall clock ]
  prob=7.525008201599121
+ idx=2
  category=426[barometer ]
  prob=4.037744522094727
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 efficientnetv2.py
```

If you want to specify the input image, put the image path after the `--input` option.  
```bash
$ python3 efficientnetv2.py --input IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 efficientnetv2.py --video VIDEO_PATH
```
The default setting is to use the optimized model and weights, but you can also switch to the normal model by using the
--normal option.

## Reference

[EfficientNetV2]( https://github.com/google/automl/tree/master/efficientnetv2 )


## Model Format

ONNX opset = 11

## Framework

Tensorflow

## Netron

[efficientnetv2-b0.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/efficientnetv2/efficientnetv2-b0.opt.onnx.prototxt)

[efficientnetv2-b1.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/efficientnetv2/efficientnetv2-b1.opt.onnx.prototxt)

[efficientnetv2-b2.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/efficientnetv2/efficientnetv2-b2.opt.onnx.prototxt)

[efficientnetv2-b3.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/efficientnetv2/efficientnetv2-b3.opt.onnx.prototxt)

