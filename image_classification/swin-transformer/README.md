# Swin Transformer for Image Classification

## Input

![Input](input.jpg)

(from https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000)

Shape : (1,3,224,224)

## Output

```
class_count=3
+ idx=0
  category=981[ballplayer, baseball player ]
  prob=8.67708683013916
+ idx=1
  category=615[knee pad ]
  prob=5.741599082946777
+ idx=2
  category=880[unicycle, monocycle ]
  prob=5.625770568847656

```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 swin_transformer.py
```

If you want to specify the input image, put the image path after the `--input` option.  
```bash
$ python3 swin_transformer.py --input IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 swin_transformer.py --video VIDEO_PATH
```


## Reference
[Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030.pdf)

[Swin Transformer](https://github.com/microsoft/Swin-Transformer)

[IMAGENET](https://image-net.org/)

[ImageNet 1000 (mini)](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000)


## Framework
Pytorch

## Model Format
ONNX opset = 11

## Netron

[swin-transformer_tiny_patch4_window7_224.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/swin-transformer/swin-transformer_tiny_patch4_window7_224.onnx.prototxt)
