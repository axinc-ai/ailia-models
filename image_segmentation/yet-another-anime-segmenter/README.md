# Yet-Another-Anime-Segmenter

## Input

<img src="anime_character.jpg" width="320px">

<!-- (Image from https://pixabay.com/photos/person-human-male-face-man-view-829966/) -->

- ailia input shape: (1, 3, ?, ?) RGB channel order
<!-- - Pixel value range: [-1, 1] -->

## Output

<img src="output.jpg" width="320px">


<!-- - ailia Predict API output:
  - Bounding boxes and keypoints
    - Shape: (1, 896, 16)
  - Classification confidences
    - Shape: (1, 896, 1)
- With helper functions, filtered detections with keypoints can be obtained. -->

## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 yet-another-anime-segmenter.py 
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 yet-another-anime-segmenter.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 yet-another-anime-segmenter.py --video VIDEO_PATH --savepath SAVE_VIDEO_PATH
```

## Reference

- [Yet-Another-Anime-Segmenter](https://github.com/zymk9/Yet-Another-Anime-Segmenter)

## Framework

PyTorch 1.7.1


## Model Format

ONNX opset = 11

## Netron

<!-- [iris.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/mediapipe_iris/iris.onnx.prototxt) -->
