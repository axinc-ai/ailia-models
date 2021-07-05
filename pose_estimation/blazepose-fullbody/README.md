# BlazePose Full Body

## Input

![Input](girl-5204299_640.jpg)

(Image from https://pixabay.com/ja/photos/%E5%A5%B3%E3%81%AE%E5%AD%90-%E7%BE%8E%E3%81%97%E3%81%84-%E8%8B%A5%E3%81%84-%E3%83%9B%E3%83%AF%E3%82%A4%E3%83%88-5204299/)

- input shape: (1, 256, 256, 3)

## Output

<img src="output.png" width="320px">

## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 blazepose-fullbody.py 
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 blazepose-fullbody.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 blazepose-fullbody.py --video VIDEO_PATH --savepath SAVE_VIDEO_PATH
```

By adding the `--model` option, you can specify model type which is selected from "lite", "full", "heavy".  
(default is full)
```bash
$ python3 blazepose-fullbody.py --model full
```

## Reference

[MediaPipe](https://google.github.io/mediapipe/solutions/models.html#pose)

## Framework

TensorFlow Lite

## Model Format

ONNX opset = 11

## Netron

[pose_detection.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/blazepose-fullbody/pose_detection.onnx.prototxt)  
[pose_landmark_lite.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/blazepose-fullbody/pose_landmark_lite.onnx.prototxt)  
[pose_landmark_full.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/blazepose-fullbody/pose_landmark_full.onnx.prototxt)  
[pose_landmark_heavy.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/blazepose-fullbody/pose_landmark_heavy.onnx.prototxt)
