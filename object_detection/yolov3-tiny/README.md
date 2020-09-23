# yolov3-tiny

## Input

![Input](couple.jpg)

- Shape : (1, 3, 416, 416)  
- Range : [0.0, 1.0]

## Output

![Output](output.png)

- category : [0,79]
- probablity : [0.0,1.0]
- position : x, y, w, h [0,1]

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 yolov3-tiny.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 yolov3-tiny.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 yolov3-tiny.py --video VIDEO_PATH
```


## Reference

- [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/)
- [A Keras implementation of YOLOv3 (Tensorflow backend)](https://github.com/qqwweee/keras-yolo3)
- [keras-onnx](https://github.com/onnx/keras-onnx/tree/master/applications/yolov3)

## Framework

Keras 2.2.4

## Model Format

ONNX opset=10

## Netron

[yolov3-tiny.opt.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/yolov3-tiny/yolov3-tiny.opt.onnx.prototxt)
