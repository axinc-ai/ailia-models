# Cascaded Pyramid Network for FashionAI Key Points Detection

## Input

### blouse
![blouse/5ac2a09a11b0488bf1e39713f36e88d4.jpg](blouse.jpg)
### dress
![dress/303e65590dc524f8eb67936bea48d489.jpg](dress.jpg)
### outwear
![outwear/513816cb9c691bef7e0edb40468717b1.jpg](outwear.jpg)
### skirt
![skirt/0ecf970028d7a6a98002c826a76f9fb1.jpg](skirt.jpg)
### trousers
![trousers/11ce236c6d8ccc54874a5d7dfdf1d8c4.jpg](trousers.jpg)

(Image from https://tianchi.aliyun.com/museum7/?spm=5176.14046517.J_9711814210.23.2bd17c0aFQzXFg#/newprodetail?productId=7)

## Output

### blouse
![Output](output_blouse.png)
### dress
![Output](output_dress.png)
### outwear
![Output](output_outwear.png)
### skirt
![Output](output_skirt.png)
### trousers
![Output](output_trousers.png)

## Example keypoints

Example keypoints of the five clothing categories are as follows.

![Example keypoints](outline.jpg)

(Image from https://github.com/HiKapok/tf.fashionAI/blob/master/demos/outline.jpg)

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 fashionai.py
```

You can specify the "clothing type" by specifying after the `--clothing-type` option.
The clothing type is selected from blouse, dress, outwear.  
```bash
$ python3 fashionai.py --clothing-type blouse
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

Pytorch

## Model Format

ONNX opset = 11

## Netron

- [blouse_100.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/fashionai-key-points-detection/blouse_100.onnx.prototxt)
- [dress_100.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/fashionai-key-points-detection/dress_100.onnx.prototxt)
- [outwear_100.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/fashionai-key-points-detection/outwear_100.onnx.prototxt)
- [skirt_100.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/fashionai-key-points-detection/skirt_100.onnx.prototxt)
- [trousers_100.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/fashionai-key-points-detection/trousers_100.onnx.prototxt)
