# VGG16

## Input

![Input](pizza.jpg)

Shape : (1,3,224,224)

## Output

```
class_count=5
+ idx=0
  category=963[pizza, pizza pie ]
  prob=16.62217140197754
+ idx=1
  category=927[trifle ]
  prob=13.598368644714355
+ idx=2
  category=926[hot pot, hotpot ]
  prob=11.545639038085938
+ idx=3
  category=567[frying pan, frypan, skillet ]
  prob=11.50587272644043
+ idx=4
  category=941[acorn squash ]
  prob=11.298295021057129
Script finished successfully.
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 vgg16.py
```

If you want to specify the input image, put the image path after the `--input` option.  
```bash
$ python3 vgg16.py --input IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 vgg16.py --video VIDEO_PATH
```


## Reference

[Very Deep Convolutional Networks for Large-Scale Image Recognition]( https://arxiv.org/abs/1409.1556 )

[Keras Applications : VGG16]( https://keras.io/applications/#vgg16 )

[keras2caffe]( https://github.com/uhfband/keras2caffe)

## Model Format

CaffeModel

## Framework

Keras

## Netron

[vgg16_pytorch.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/vgg16/vgg16_pytorch.onnx.prototxt)
