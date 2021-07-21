# EfficientDet

## Input

![Input](img.png)

(Image from https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/master/test/img.png)

## Output

![Output](output.png)

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 efficientdet.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 efficientdet.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 efficientdet.py --video VIDEO_PATH
```

By adding the model name after the `--model` option, you can specify the model.  
The model name is selected from 'd0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd0hd', 'd1hd', 'd2hd', 'd3hd', 'd4hd'.
```bash
$ python3 efficientdet.py --model d0hd
```

## Reference

- [Yet Another EfficientDet Pytorch Convert ONNX TVM](https://github.com/murdockhou/Yet-Another-EfficientDet-Pytorch-Convert-ONNX-TVM)
- [EfficientDet](https://github.com/google/automl/tree/master/efficientdet)
- [EfficientDet: Scalable and Efficient Object Detection, in PyTorch](https://github.com/toandaominh1997/EfficientDet.Pytorch)
- [Yet Another EfficientDet Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[efficientdet-d0.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/efficientdet/efficientdet-d0.onnx.prototxt)
[efficientdet-d1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/efficientdet/efficientdet-d1.onnx.prototxt)
[efficientdet-d2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/efficientdet/efficientdet-d2.onnx.prototxt)
[efficientdet-d3.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/efficientdet/efficientdet-d3.onnx.prototxt)
[efficientdet-d4.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/efficientdet/efficientdet-d4.onnx.prototxt)
[efficientdet-d5.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/efficientdet/efficientdet-d5.onnx.prototxt)
[efficientdet-d6.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/efficientdet/efficientdet-d6.onnx.prototxt)
[efficientdet-d0hd.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/efficientdet/efficientdet-d0hd.onnx.prototxt)
[efficientdet-d1hd.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/efficientdet/efficientdet-d1hd.onnx.prototxt)
[efficientdet-d2hd.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/efficientdet/efficientdet-d2hd.onnx.prototxt)
[efficientdet-d3hd.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/efficientdet/efficientdet-d3hd.onnx.prototxt)
[efficientdet-d4hd.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/efficientdet/efficientdet-d4hd.onnx.prototxt)