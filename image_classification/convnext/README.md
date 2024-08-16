# ConvNeXt

## Input

![Input](input_imagenet.jpg)

Ailia input shape : (1,3,224,224)  
Range : [-1.0, 1.0]

## Output
If specified model is `base_1k`, `small_1k` or `tiny_1k`, it predicts image class from `label_table.txt`.
```
predicted class = 981(ballplayer, baseball player)
```
If specified model is `cifar10`, it predicts image class from `['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']`.
```
predicted class = 1(automobile)
```

## Usage

Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet while downloading.

For the sample image,
```
$ python3 convnext.py
```
If you want to specify the input image, put the image path after the `--input` option.
```
$ python3 convnext.py --input IMAGE_PATH
```
By adding the `--video` option, you can input the video.
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```
$ python3 convnext.py --video VIDEO_PATH
```

By adding the `--model` option, you can choose model.

```
$ python3 convnext.py --input IMAGE_PATH --model MODEL_TYPE

(ex)$ python3 convnext.py --input IMAGE_PATH --model base_1k

(ex)$ python3 convnext.py --input IMAGE_PATH --model small_1k

(ex)$ python3 convnext.py --input IMAGE_PATH --model tiny_1k

(ex)$ python3 convnext.py --input IMAGE_PATH --model cifar10
```

## Reference

[A PyTorch implementation of ConvNeXt](https://github.com/facebookresearch/ConvNeXt)

[IMAGENET](https://image-net.org/)

[ImageNet 1000 (mini)](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000)

[The CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

[CIFAR-10-images(Github)](https://github.com/YoongiKim/CIFAR-10-images)

## Model Format

ONNX opset = 10

## Framework

Pytorch 1.7.1

## Netron

[convnext_base_1k_224_ema.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/convnext/convnext_base_1k_224_ema.onnx.prototxt)

[convnext_small_1k_224_ema.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/convnext/convnext_small_1k_224_ema.onnx.prototxt)

[convnext_tiny_1k_224_ema.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/convnext/convnext_tiny_1k_224_ema.onnx.prototxt)  

[convnext_tiny_CIFAR-10.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/convnext/convnext_tiny_CIFAR-10.onnx.prototxt)
