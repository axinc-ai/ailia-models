# Crowd Counting Code Framework (C3-Framework)

## Input

![Input](demo.jpg)

(Image from https://www.kaggle.com/tthien/shanghaitech)

Shape : (batch, 3, height, width)

## Output (the desity maps)

![Output](output.png)

Shape : (batch, 1, height, width)

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 c3.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 c3.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 c3.py --video VIDEO_PATH
```

By adding the model name after the `--model` option, you can specify the model.  
The model name is selected from 'alexnet', 'vgg', 'vgg_decoder', 'resnet50', 'resnet101', 'csrnet', 'sanet'.
```bash
$ python3 c3.py --model alexnet
```

## Reference

- [C-3-Framework](https://github.com/gjy3035/C-3-Framework)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[AlexNet.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/c-3-framework/AlexNet.onnx.prototxt)
[VGG.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/c-3-framework/VGG.onnx.prototxt)
[VGG_DECODER.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/c-3-framework/VGG_DECODER.onnx.prototxt)
[ResNet50.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/c-3-framework/ResNet50.onnx.prototxt)
[ResNet101.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/c-3-framework/ResNet101.onnx.prototxt)
[CSRNet.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/c-3-framework/CSRNet.onnx.prototxt)
[SANet.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/c-3-framework/SANet.onnx.prototxt)
