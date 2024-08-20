# PP-PicoDet

## Input

![Input](demo.jpg)

(Image from https://cocodataset.org/#download)

## Output

![Output](output.png)

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 picodet.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 picodet.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 picodet.py --video VIDEO_PATH
```

By adding the `--model_type` option, you can specify model type which is selected from "s-416", "m-416", "l-640". (default is s-416)
```bash
$ python3 picodet.py --model_type s-416
```

## Reference

- [PP-PicoDet](https://github.com/Bo396543018/Picodet_Pytorch)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[picodet_s_416_coco.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/picodet/picodet_s_416_coco.onnx.prototxt)  
[picodet_m_416_coco.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/picodet/picodet_m_416_coco.onnx.prototxt)  
[picodet_l_640_coco.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/picodet/picodet_l_640_coco.onnx.prototxt)
