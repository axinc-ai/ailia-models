# YOLOV6

## Input

![Input](input.jpg)

(Image from https://github.com/RangiLyu/nanodet/blob/main/demo_mnn/imgs/000252.jpg)

Ailia input shape: (1, 3, 640, 640)

## Output

![Output](output.jpg)

## Usage

Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet
while downloading.

For the sample image,

``` bash
$ python3 yolov6.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.

```bash
$ python3 yolov6.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.

```bash
$ python3 yolov6.py --video VIDEO_PATH
```

You can use `--model_name` option to change the model architecture.
```bash
$ python3 yolovv6.py --model_name yolov6t
```

You can use `--detection_width` and `--detection_height` options to change the detection resolution
```bash
$ python3 yolov6.py --detection_width 640 --detection_height 640
```

## Reference

[YOLOV6](https://github.com/meituan/YOLOv6)

## Framework

Pytorch

## Model Format

ONNX opset = 12

## Netron

[yolov6n.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolov6/yolov6n.onnx.prototxt)

[yolov6s.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolov6/yolov6s.onnx.prototxt)

[yolov6t.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolov6/yolov6t.onnx.prototxt)