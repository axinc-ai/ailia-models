# Weather Prediction From Image

### input
![input_image](https://user-images.githubusercontent.com/78332175/120158006-99d3a280-c22e-11eb-96c7-c70c05dacc5f.jpg)

Ailia input shape: (1, 100, 100, 3)  


### output
```
class_count=5
+ idx=0
  category=1[Sunny ]
  prob=0.89111328125
+ idx=1
  category=0[Cloudy ]
  prob=0.10894775390625
+ idx=2
  category=3[Snowy ]
  prob=5.960464477539063e-08
+ idx=3
  category=4[Foggy ]
  prob=0.0
+ idx=4
  category=2[Rainy ]
  prob=0.0
```

### usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 WeatherPredictionFromImage.py
```

If you want to specify the input image, put the image path after the `--input` option.  
```bash
$ python3 WeatherPredictionFromImage.py --input IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 WeatherPredictionFromImage.py --video VIDEO_PATH
```

### Reference
[Weather Prediction From Image - (Warmth Of Image)](https://github.com/berkgulay/WeatherPredictionFromImage)

### Framework

tensorflow.keras (tensorflow==2.0.0)

### Model Format
ONNX opset = 11

### Netron

[wpfi.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/WeatherPredictionFromImage/wpfi.onnx.prototxt)

