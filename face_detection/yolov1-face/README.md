# yolov1-face

## Input

![Input](couple.jpg)

## Output

![Output](output.png)

- probablity : [0.0,1.0]
- position : x, y, w, h [0,1]

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 yolov1-face.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 yolov1-face.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 yolov1-face.py --video VIDEO_PATH
```


## Reference

- [YOLO-Face-detection](https://github.com/dannyblueliu/YOLO-Face-detection)
- [convert between pytorch, caffe prototxt/weights and darknet cfg/weights](https://github.com/marvis/pytorch-caffe-darknet-convert)

## Framework

Darknet

## Model Format

CaffeModel

## Netron

[yolov1-face.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolov1-face/yolov1-face.prototxt)
