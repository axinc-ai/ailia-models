# YOLOv11

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
$ python3 yolov11.py
```

If you want to specify the input image, put the image path after the `--input` option.
You can use `--savepath` option to change the name of the output file to save.

```bash
$ python3 yolov11.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.

```bash
$ python3 yolov11.py --video VIDEO_PATH
```

By adding the `--model_type` option, you can specify model type which is selected from "v11n", "v11s" , "v11m", "v11l", "v11x"(default is "v11x").

```bash
$ python3 yolov11.py --model_type v11x
```

You can use `--detection_size` option to change the detection resolution.

```bash
$ python3 yolov11.py --detection_size 1280
```

## Reference

- [YOLOv11](https://github.com/ultralytics/ultralytics)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

- [yolov11n.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolov11/yolov11n.onnx.prototxt)
- [yolov11s.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolov11/yolov11s.onnx.prototxt)
- [yolov11m.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolov11/yolov11m.onnx.prototxt)
- [yolov11l.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolov11/yolov11l.onnx.prototxt)
- [yolov11x.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolov11/yolov11x.onnx.prototxt)
