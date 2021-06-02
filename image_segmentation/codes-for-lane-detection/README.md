# Codes-for-Lane-Detection

## Input

![Input](input.jpg)

(Image from https://github.com/czming/RONELD-Lane-Detection/tree/main/example/00000.jpg)

Input shape: (1, 3, 208, 976) for erfnet, (1, 288, 800, 3) for scnn

## Output

![Output](output.jpg)

## Usage

Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet
while downloading.

For the sample image,

``` bash
$ python3 codes-for-lane-detection.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.

```bash
$ python3 codes-for-lane-detection.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.

```bash
$ python3 codes-for-lane-detection.py --video VIDEO_PATH
```

By adding the `--arch` option, you can select the model architecture from `erfnet` and `scnn`. 

## Reference

[Codes-for-Lane-Detection](https://github.com/cardwing/Codes-for-Lane-Detection/)

[ERFNet-CULane-PyTorch](https://github.com/cardwing/Codes-for-Lane-Detection/tree/master/ERFNet-CULane-PyTorch)

[Spatial As Deep: Spatial CNN for Traffic Scene Understanding](https://github.com/cardwing/Codes-for-Lane-Detection/tree/master/SCNN-Tensorflow)

## Framework

ERFNet : Pytorch

SCNN : Tensorflow 1.13.2


## Model Format

ONNX opset = 11

## Netron

[erfnet.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/codes-for-lane-detection/erfnet.opt.onnx.prototxt)