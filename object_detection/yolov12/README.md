# YOLOv12

## Input

![Input](input.jpg)

(Image from https://pixabay.com/ja/photos/%E3%83%AD%E3%83%B3%E3%83%89%E3%83%B3%E5%B8%82-%E9%8A%80%E8%A1%8C-%E3%83%AD%E3%83%B3%E3%83%89%E3%83%B3-4481399/)

Ailia input shape : (1, 3, h, w)
Range : [0.0, 1.0]
Color : RGB

## Output

![Output](output.png)

## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,

```bash
$ python3 yolov12.py
```

If you want to specify the input image, put the image path after the `--input` option.
You can use `--savepath` option to change the name of the output file to save.

```bash
$ python3 yolov12.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.

```bash
$ python3 yolov12.py --video VIDEO_PATH
```

By adding the `--model_type` option, you can specify model type which is selected from "v12n", "v12s" , "v12m", "v12l", "v12x"(default is "v12x").

```bash
$ python3 yolov12.py --model_type v12x
```

You can use `--detection_size` option to change the detection resolution.

```bash
$ python3 yolov12.py --detection_size 1280
```

## Reference

- [YOLOv12](https://github.com/sunsmarterjie/yolov12)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

- [yolov12n.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolov12/yolov12n.onnx.prototxt)
- [yolov12s.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolov12/yolov12s.onnx.prototxt)
- [yolov12m.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolov12/yolov12m.onnx.prototxt)
- [yolov12l.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolov12/yolov12l.onnx.prototxt)
- [yolov12x.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolov12/yolov12x.onnx.prototxt)
