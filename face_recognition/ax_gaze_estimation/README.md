# ax Gaze Estimation

## Input

<img src="woman_face.jpg" width="320px">

(Image from https://pixabay.com/photos/vintage-woman-hat-fashion-style-635244/)

### Face Detector: BlazeFace

- ailia input shape: (1, 3, 128, 128) RGB channel order
- Pixel value range: [-1, 1]

### Face Landmarks Estimator: Face Mesh

- ailia input shape: (batch_size, 3, 192, 192) RGB channel order
- Pixel value range: [-1, 1]

### Iris Landmarks Estimator (Optional): MediaPipe Iris

- ailia input shape: (batch_size, 3, 64, 64) RGB channel order
- Pixel value range: [-1, 1]
- Left eye (or horizontally flipped right eye)

### Head Pose Estimator (Optional): Hopenet / Hopenet Lite

- ailia input shape: (batch_size, 3, 224, 224) RGB channel order
- Pixel value range: [0, 1] before normalization
- Preprocessing: normalization using ImageNet statistics

### Gaze Estimator: ax Gaze Estimation

- ailia input shape:
  - face image(s): (batch_size, 192, 192, 3) RGB channel order
  - head pose(s) (optional): (batch_size, 3) roll (left+), yaw (right+), pitch (down+) in radians
- Pixel value range: [-1, 1]

## Output

<img src="output.png" width="320px">

### Face Detector: BlazeFace

- ailia Predict API output:
  - Bounding boxes and keypoints
    - Shape: (1, 896, 16)
  - Classification confidences
    - Shape: (1, 896, 1)
- With helper functions, filtered detections with keypoints can be obtained.

### Face Landmarks Estimator: Face Mesh

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
- With helper functions, image (original size) coordinates of eye centers, iris landmarks and
cropped eye region image can be obtained.

### Iris Landmarks Estimator (Optional): MediaPipe Iris

- ailia Predict API output:
  - `eyes`: 71 eye/eyebrow region landmarks with (x, y, z) coordinates
    - Shape: (1, 213 * batch_size)
    - x and y are in the range [0, 64] (origin is upper left corner for both
    left and FLIPPED right eye image). z is unused (refer to Face Estimator
    for those interested in using this coordinate).
    - The 16 points defining one eye's contour is refined with this estimator
    (16 first points).
  - `iris`: 5 iris landmarks with (x, y, z) coordinates
    - Shape: (1, 15 * batch_size)
    - x and y are in the range [0, 64] (origin is upper left corner for both
    left and FLIPPED right eye image). z coordinate is set to the average of the
    z coordinate of the eye corners.
- With helper functions, image (original size) coordinates of iris landmarks can
be obtained.

### Head Pose Estimator (Optional): Hopenet / Hopenet Lite

- ailia Predict API output:
  - `yaw`: scores for yaw angle
    - Shape: (batch_size, 66)
  - `pitch`: scores for pitch angle
    - Shape: (batch_size, 66)
  - `roll`: scores for roll angle
    - Shape: (batch_size, 66)
- With helper functions, yaw, pitch and roll in radians can be obtained.

### Gaze Estimator: ax Gaze Estimation

- ailia Predict API output:
  - `gazes`: (phi, theta) angles in spherical coordinates
    - Shape: (batch_size, 2)
- With helper functions, 3D gaze vectors can be obtained.

## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 ax_gaze_estimation.py 
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 ax_gaze_estimation.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 ax_gaze_estimation.py --video VIDEO_PATH --savepath SAVE_VIDEO_PATH
```

#### Additional options

- `--include-iris`: Perform iris landmarks estimation or not.
- `--draw-iris`: Draw the iris landmarks or not.
- `--include-head-pose`: Perform head pose estimation or not.
- `--draw-head-pose`: Draw the head pose(s) or not.
- `-l` or `--lite`: Use a lite version of Hopenet.

## Model Format

ONNX opset = 10
