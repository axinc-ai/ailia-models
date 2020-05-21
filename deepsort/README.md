# DeepSort



### Input
- DeepSORT original mode:
  - The sample input video is `TownCentreXVID.avi` from http://www.robots.ox.ac.uk/~lav/Research/Projects/2009bbenfold_headpose/project.html

- compare image mode: If two input images have the same person or not ?
  - Our detector, yolo, detects the person and resizes the cropped image automatically, 
  so the size of the input images are not specified.
  However, please note that at this stage, it is assumed that only one person is in one image.  
  - The sample input images are downloaded from [MARS dataset](http://www.liangzheng.com.cn/Project/project_mars.html)
  - correct pair  
  ![correct_pair_1](correct_32_1.jpg)
  ![correct_pair_2](correct_32_2.jpg)
  
  - not correct pair  
  ![correct_pair_1](correct_32_1.jpg)
  ![false_pair_1](false_14_1.jpg)

### Output Example
- DeepSORT original mode:
  ![ex_result](https://user-images.githubusercontent.com/45060776/82342446-978c1500-9a2c-11ea-976b-a0d3358f89a3.gif)

- compare image mode:
  ```bash
  ['correct_32_1.jpg', 'correct_32_2.jpg']: SAME person (confidence: 0.8727510571479797)
  ['correct_32_1.jpg', 'false_14_1.jpg']: Diefferent person (confidence: 0.6612733006477356)
  ```

### Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

- DeepSORT original mode:
  For the sample video,
  ``` bash
  $ python3 deepsort.py
  ```

  If you want to specify the input video, put the video path after the `--video` option.  
  You can use `--savepath` option to change the name of the output file to save.  
  If you pass `0` as an argument to `VIDEO_PATH`, you can use the webcam input instead of the video.
  ```bash
  $ python3 deepsort.py --video VIDEO_PATH --savepath SAVE_IMAGE_PATH
  ```

- compare image mode:
  The -p option must be followed by the paths of the two images you want to compare.   
  Please note that it is assumed that one person is in one image.  
  ```bash
  $ python3 deepsort.py --pairimage IMAGE_PATH1 IMAGE_PATH2
  ```

### Reference

[Deep Sort with PyTorch](https://github.com/ZQPei/deep_sort_pytorch)


### Framework
PyTorch 0.4


### Model Format
ONNX opset = 10


### Netron
[deep_sort.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/deep_sort/deep_sort.onnx.prototxt)
