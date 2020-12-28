# Face Mesh

## Input

<img src="man.jpg" width="320px">

(Image from https://pixabay.com/photos/person-human-male-face-man-view-829966/)

### Detector

- ailia input shape: (1, 3, 128, 128) RGB channel order
- Pixel value range: [-1, 1]

### Landmark

- ailia input shape: (batch_size, 3, 192, 192) RGB channel order
- Pixel value range: [-1, 1]

## Output

<img src="output.png" width="320px">

### Detector

- ailia Predict API output:
  - Bounding boxes and keypoints
    - Shape: (1, 896, 16)
  - Classification confidences
    - Shape: (1, 896, 1)
- With helper functions, filtered detections with keypoints can be obtained.

### Estimator

- ailia Predict API output:
  - `landmarks`: 468 face landmarks with (x, y, z) coordinates
    - Shape: (batch_size, 468, 3)
    - x and y are in the range [0, 192] (to normalize, divide by the image width
    and height, 192). z represents the landmark depth with the depth at center
    of the head being the origin, and the smaller the value the closer the
    landmark is to the camera. The magnitude of z uses roughly the same scale as
    x.
  - `confidences`: no information. Probably a confidence score for the landmarks
    - Shape: (batch_size,)
- With helper functions, image coordinates of hand landmarks can be obtained.

## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 facemesh.py 
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 facemesh.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 facemesh.py --video VIDEO_PATH --savepath SAVE_VIDEO_PATH
```

## Reference

- [facemesh.pytorch](https://github.com/thepowerfuldeez/facemesh.pytorch)
- [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh)

## Framework

PyTorch 1.7.1


## Model Format

ONNX opset = 10

## Netron

[facemesh.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/facemesh/facemesh.onnx.prototxt)


