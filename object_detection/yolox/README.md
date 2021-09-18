# YOLOX

## Input

![Input](input.jpg)

(Image from https://github.com/RangiLyu/nanodet/blob/main/demo_mnn/imgs/000252.jpg)

Ailia input shape: (1, 3, 416, 416), (1, 3, 640, 640)

## Output

![Output](output.jpg)

## Usage

Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet
while downloading.

For the sample image,

``` bash
$ python3 yolox.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.

```bash
$ python3 yolox.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.

```bash
$ python3 yolox.py --video VIDEO_PATH
```

The default setting is to use the optimized model and weights, but you can also switch to the normal model by using the
--normal option.

## Reference

[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)

## Framework

Pytorch

## Model Format

ONNX opset = 11

## Netron

[yolox_nano.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolox/yolox_nano.opt.onnx.prototxt)

[yolox_tiny.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolox/yolox_tiny.opt.onnx.prototxt)

[yolox_s.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolox/yolox_s.opt.onnx.prototxt)

[yolox_m.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolox/yolox_m.opt.onnx.prototxt)

[yolox_l.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolox/yolox_l.opt.onnx.prototxt)

[yolox_darknet.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolox/yolox_darknet.opt.onnx.prototxt)

[yolox_x.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolox/yolox_x.opt.onnx.prototxt)