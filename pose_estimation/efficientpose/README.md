# EfficientPose

## Input

<img src="MPII.jpg" width="320px">

(Image from https://github.com/daniegr/EfficientPose/blob/master/utils/MPII.jpg)

Model variant: RT
Ailia input shape : (1, 224, 224, 3)  
Range : [0, 1.0]

## Output

<img src="MPII_out.png" width="320px">

- Confidence : (1, 224, 224, 16)
- Range : [0, 1.0]

## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 efficientpose.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 efficientpose.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

If you want to specify the model variant, put the model variant after the `--model_variant` option.  
You can only choose variants from 'rt','i','ii','iii','iv'.
```bash
$ python3 efficientpose.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH --model_variant rt
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 efficientpose.py --video VIDEO_PATH
```

## Reference

[Code repo for EfficientPose](https://github.com/daniegr/EfficientPose)

## Framework

Keras, TensorFlow, TensorFlow Lite or PyTorch

## Model Format

ONNX opset = 10

## Netron

[EfficientPoseRT.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/efficientpose/EfficientPoseRT.onnx.prototxt)

[EfficientPoseI.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/efficientpose/EfficientPoseI.onnx.prototxt)

[EfficientPoseII.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/efficientpose/EfficientPoseII.onnx.prototxt)

[EfficientPoseIII.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/efficientpose/EfficientPoseIII.onnx.prototxt)

[EfficientPoseIV.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/efficientpose/EfficientPoseIV.onnx.prototxt)

