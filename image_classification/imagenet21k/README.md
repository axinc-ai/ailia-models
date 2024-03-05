# Imagenet21K

## Input

![Input](input.jpg)

(Image from https://github.com/pytorch/hub/raw/master/images/dog.jpg)

Ailia input shape: (1,3,299,299)
Range: [0.0, 255.0]

## Output
```
class_count=7
+ idx=0
  category=1531[spitz ]
  prob=84.49806213378906
+ idx=1
  category=1386[dog ]
  prob=81.74344635009766
+ idx=2
  category=155[domestic_animal ]
  prob=81.34649658203125
+ idx=3
  category=3[animal ]
  prob=81.01296997070312
+ idx=4
  category=1532[Samoyed ]
  prob=59.17914581298828
+ idx=5
  category=1385[bitch ]
  prob=38.642704010009766
+ idx=6
  category=1384[canine ]
  prob=37.53316879272461
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

You can select a pretrained model by specifying -a mixer, resnet50 ,mobilenet or vit(default).

## Reference

[ImageNet21K](https://github.com/Alibaba-MIIL/ImageNet21K)

## Model Format

ONNX opset = 14

## Framework

pytorch

## Netron

[mobilenetv3_large_100.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/imagenet21k/mobilenetv3_large_100.onnx.prototxt)

[mixer_b16_224_miil_in21k.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/imagenet21k/mixer_b16_224_miil_in21k.onnx.prototxt)

[resnet50.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/imagenet21k/resnet50.onnx.prototxt)

[vit_base_patch16_224_miil_in21k](https://netron.app/?url=https://storage.googleapis.com/ailia-models/imagenet21k/vit_base_patch16_224_miil_in21k)
