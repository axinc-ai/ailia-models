# DeepSort Vehicle 


## Input

- DeepSORT original mode:
  - Tracking of the vehicle in the input video.
- Compare image mode:
  - If two input images have the same vehicle or not ?
  - Our detector, yolo, detects the vehicle and resizes the cropped image automatically, 
  so the size of the input images are not specified.
  However, please note that at this stage, it is assumed that only one vehicle is in one image.  

## Output

- Compare image mode:
  ```bash
  ['01840_c40752s1_00280_01.jpg', '01840_c40752s1_00307_01.jpg']: Same vehicle (distance: 0.015761730851141575)
  ```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

- DeepSORT original mode:
  For the video,
  ``` bash
  $ python3 deepsort_vehicle.py
  ```

  If you want to specify the input video, put the video path after the `--video` option.  
  You can use `--savepath` option to change the name of the output file to save.  
  If you pass `0` as an argument to `VIDEO_PATH`, you can use the webcam input instead of the video.
  ```bash
  $ python3 deepsort_vehicle.py --video VIDEO_PATH --savepath SAVE_VIDEO_PATH
  ```

- compare image mode:
  The `-p` option must be followed by the paths of the two images you want to compare.   
  Please note that it is assumed that one vehicle is in one image.  
  ```bash
  $ python3 deepsort_vehicle.py --pairimage IMAGE_PATH1 IMAGE_PATH2
  ```

## Reference

- [Multi-Camera Live Object Tracking](https://github.com/LeonLok/Multi-Camera-Live-Object-Tracking)
- [cosine_metric_learning](https://github.com/nwojke/cosine_metric_learning)

## Framework

TensorFlow

## Model Format

ONNX opset = 11

## Netron

[deep_sort_vehicle.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/deep_sort_vehicle/deep_sort_vehicle.onnx.prototxt)
