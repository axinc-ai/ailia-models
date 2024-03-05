# VOLO

## Input

![Input](pizza.jpg)

Shape : (1,3,224,224)

## Output

```
class_count=3
+ idx=0
  category=963[pizza, pizza pie ]
  prob=12.82837963104248
+ idx=1
  category=923[plate ]
  prob=3.2662553787231445
+ idx=2
  category=572[goblet ]
  prob=2.8674607276916504
Script finished successfully.
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 volo.py
```

If you want to specify the input image, put the image path after the `--input` option.  
```bash
$ python3 volo.py --input IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 volo.py --video VIDEO_PATH
```


## Reference

[VOLO: Vision Outlooker for Visual Recognition](https://github.com/sail-sg/volo)

## Model Format

ONNX opset = 11

## Framework

Pytorch 2.2.0

## Netron

[volo_d1_224.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/volo/volo.onnx.prototxt)

[volo_d1_384.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/volo/volo.onnx.prototxt)

[volo_d2_224.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/volo/volo.onnx.prototxt)

[volo_d2_384.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/volo/volo.onnx.prototxt)

[volo_d3_224.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/volo/volo.onnx.prototxt)

[volo_d3_448.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/volo/volo.onnx.prototxt)

[volo_d4_224.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/volo/volo.onnx.prototxt)

[volo_d4_448.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/volo/volo.onnx.prototxt)

[volo_d5_224.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/volo/volo.onnx.prototxt)
