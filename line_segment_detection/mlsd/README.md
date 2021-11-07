# M-LSD: Towards Light-weight and Real-time Line Segment Detection

## Input

![Input](input.jpg)

(Image from https://pixabay.com/ja/photos/%e6%97%85%e8%a1%8c%e3%81%99%e3%82%8b-%e3%83%9b%e3%83%86%e3%83%ab%e3%81%ae%e9%83%a8%e5%b1%8b-%e3%83%9b%e3%83%86%e3%83%ab-1677347/)

Input shape: (1, 512, 512, 4)

## Output

![Output](output.jpg)

## Usage

Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet
while downloading.

For the sample image,

``` bash
$ python3 mlsd.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.

```bash
$ python3 mlsd.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.

```bash
$ python3 mlsd.py --video VIDEO_PATH
```

## Reference

[M-LSD: Towards Light-weight and Real-time Line Segment Detection](https://github.com/navervision/mlsd)

## Framework

Tensorflow

## Model Format

ONNX opset = 11

## Netron

[M-LSD_512_large.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/mlsd/M-LSD_512_large.opt.onnx.prototxt)
