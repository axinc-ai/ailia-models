# Real-time face detection and emotion/gender classification

## Input

![Input](lenna.png)

Ailia input shape: (1, 1, 64, 64)  
Range: [-1.0, 1.0]

## Output

```
emotion_class_count=3
+ idx=0
  category=6[ neutral ]
  prob=0.411855548620224
+ idx=1
  category=4[ sad ]
  prob=0.1994263231754303
+ idx=2
  category=0[ angry ]
  prob=0.19452838599681854

gender_class_count=2
+ idx=0
  category=0[ female ]
  prob=0.8007728457450867
+ idx=1
  category=1[ male ]
  prob=0.19922710955142975
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 blazeface.py 
```

If you want to specify the input image, put the image path after the `--input` option.  
```bash
$ python3 blazeface.py --input IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 blazeface.py --video VIDEO_PATH
```


## Reference

[Real-time face detection and emotion/gender classification](https://github.com/oarriaga/face_classification)

## Framework

Keras

## Model Format

CaffeModel

## Netron

[emotion_miniXception.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/face_classification/emotion_miniXception.prototxt)
