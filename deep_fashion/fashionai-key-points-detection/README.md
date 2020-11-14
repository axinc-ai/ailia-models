# Cascaded Pyramid Network for FashionAI Key Points Detection

## Input

![fashionAI_keypoints_train1.tar/Images/dress/303e65590dc524f8eb67936bea48d489.jpg](dress.jpg)

(Image from https://tianchi.aliyun.com/museum7/?spm=5176.14046517.J_9711814210.23.2bd17c0aFQzXFg#/newprodetail?productId=7)

## Output

![Output](output_dress.png)

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 fashionai.py
```

If you want to specify the input image, put the image path after the `--input` option.  
```bash
$ python3 fashionai.py --input IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 fashionai.py --video VIDEO_PATH
```

## Reference

- [A Pytorch Implementation of Cascaded Pyramid Network for FashionAI Key Points Detection](https://github.com/gathierry/FashionAI-KeyPointsDetectionOfApparel)

## Framework

ONNX Runtime

## Model Format

ONNX opset = 11

## Netron

- [dress_100.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/fashionai-key-points-detection/dress_100.onnx.prototxt)
