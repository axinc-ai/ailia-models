# FastestDet

## Input

![Input](demo.jpg)

(Image from https://github.com/dog-qiuqiu/FastestDet/tree/main/data)

## Output

![Output](output.png)

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 fastest-det.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 fastest-det.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 fastest-det.py --video VIDEO_PATH
```

## Reference

- [FastestDet](https://github.com/dog-qiuqiu/FastestDet)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[FastestDet.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/fastest-det/FastestDet.onnx.prototxt)
