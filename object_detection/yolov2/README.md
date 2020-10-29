# yolov2

## Input

![Input](couple.jpg)

Shape : (1, 3, 416, 416)  
Range : [-1.0, 1.0]

## Output

![Output](output.png)

- category : [0,79]  
- probablity : [0.0,1.0]  
- position : x, y, w, h [0,1]  


### usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 yolov2.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 yolov2.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 yolov2.py --video VIDEO_PATH
```

## Reference

- [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolov2/)
- [Covert original YOLO model from Pytorch to Onnx, and do inference using backend Caffe2 or Tensorflow.](https://github.com/purelyvivid/yolo2_onnx)

## Framework

Pytorch 1.3.1

## Model Format

ONNX opset=10

## Netron

[yolov2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolov2/yolov2.onnx.prototxt)
