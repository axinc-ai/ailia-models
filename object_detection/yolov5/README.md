# yolov5

## Input

![Input](bus.jpg)

(Image from https://github.com/ultralytics/yolov5/blob/master/data/images/bus.jpg)

Shape : (1, 3, 640, 640)  
Range : [0.0, 1.0]
Color : RGB

## Output

![Output](output.png)

## usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 yolov5.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 yolov5.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 yolov5.py --video VIDEO_PATH
```

## Reference

- [yolov5](https://github.com/ultralytics/yolov5)

## Framework

ONNX Runtime

## Model Format

ONNX opset=11

## Netron

[yolov5s.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolov5/yolov5s.onnx.prototxt)
[yolov5m.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolov5/yolov5m.onnx.prototxt)
[yolov5l.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolov5/yolov5l.onnx.prototxt)
[yolov5x.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolov5/yolov5x.onnx.prototxt)
