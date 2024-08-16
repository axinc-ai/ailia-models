# yolov5

## Input

![Input](input.jpg)

(Image from https://pixabay.com/ja/photos/%E3%83%AD%E3%83%B3%E3%83%89%E3%83%B3%E5%B8%82-%E9%8A%80%E8%A1%8C-%E3%83%AD%E3%83%B3%E3%83%89%E3%83%B3-4481399/)

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

You can use `--arch` option to change the model architecture.
```bash
$ python3 yolov5.py --arch yolov5m
```

You can use `--detection_width` and `--detection_height` options to change the detection resolution
```bash
$ python3 yolov5.py --detection_width 1280 --detection_height 640
```

## Reference

- [yolov5](https://github.com/ultralytics/yolov5)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[yolov5s.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolov5/yolov5s.onnx.prototxt)

[yolov5m.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolov5/yolov5m.onnx.prototxt)

[yolov5l.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolov5/yolov5l.onnx.prototxt)

[yolov5x.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolov5/yolov5x.onnx.prototxt)

[yolov5s6.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolov5/yolov5s6.onnx.prototxt)
