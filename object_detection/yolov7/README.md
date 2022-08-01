# YOLOv7

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
$ python3 yolov7.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.

```bash
$ python3 yolov7.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.

```bash
$ python3 yolov7.py --video VIDEO_PATH
```

The default setting is to use the optimized model and weights, but you can also switch to the normal model by using the
--normal option.

## Reference

[YOLOv7](https://github.com/WongKinYiu/yolov7)

## Framework

Pytorch

## Model Format

ONNX opset = 11

## Netron

[yolov7.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolov7/yolov7.opt.onnx.prototxt)

[yolov7x.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolov7/yolov7x.opt.onnx.prototxt)

[yolov7_w6.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolov7/yolov7_w6.opt.onnx.prototxt)

[yolov7_e6.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolov7/yolov7_e6.opt.onnx.prototxt)

[yolov7_d6.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolov7/yolov7_d6.opt.onnx.prototxt)

[yolov7_e6e.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolov7/yolov7_e6e.opt.onnx.prototxt)