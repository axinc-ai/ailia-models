# Hopenet

## Input

<img src="man.jpg" width="320px">

(Image from https://pixabay.com/photos/person-human-male-face-man-view-829966/)

### Face Detector: BlazeFace

- ailia input shape: (1, 3, 128, 128) RGB channel order
- Pixel value range: [-1, 1]

### Head Pose Estimator: Hopenet / Hopenet Lite

- ailia input shape: (batch_size, 3, 224, 224) RGB channel order
- Pixel value range: [0, 1] before normalization
- Preprocessing: normalization using ImageNet statistics

## Output

<img src="output.png" width="320px">

### Face Detector: BlazeFace

- ailia Predict API output:
  - Bounding boxes and keypoints
    - Shape: (1, 896, 16)
  - Classification confidences
    - Shape: (1, 896, 1)
- With helper functions, filtered detections with keypoints can be obtained.

### Head Pose Estimator: Hopenet / Hopenet Lite

- ailia Predict API output:
  - `yaw`: scores for yaw angle
    - Shape: (batch_size, 66)
  - `pitch`: scores for pitch angle
    - Shape: (batch_size, 66)
  - `roll`: scores for roll angle
    - Shape: (batch_size, 66)
- With helper functions, yaw, pitch and roll in radians can be obtained.

## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 hopenet.py 
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 hopenet.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 hopenet.py --video VIDEO_PATH --savepath SAVE_VIDEO_PATH
```

By adding the `--lite` option, a lite version of Hopenet is used.
```bash
$ python3 blazehand.py --lite
```

## Reference

- [deep-head-pose](https://github.com/natanielruiz/deep-head-pose)
- [deep-head-pose-lite](https://github.com/OverEuro/deep-head-pose-lite)

## Framework

PyTorch 1.7.1

## Model Format

ONNX opset = 10

## Netron

- [hopenet_robust_alpha1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/hopenet/hopenet_robust_alpha1.onnx.prototxt)
- [hopenet_lite.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/hopenet/hopenet_lite.onnx.prototxt)
