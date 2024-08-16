# mobile_object_localizer_v1

## Input

![Input](demo.jpg)

(Image from https://commons.wikimedia.org/wiki/File:Il_cuore_di_Como.jpg)

Shape : (1, 3, 192, 192)  

## Output

![Output](output.png)

- detection_boxes shape : (1, 100, 4)
- detection_classes shape : (1, 100)
- detection_scores shape : (1, 100)
- num_detections shape : (1,)

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 mobile_object_localizer.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 mobile_object_localizer.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 mobile_object_localizer.py --video VIDEO_PATH
```

## Reference

- [mobile_object_localizer_v1](https://tfhub.dev/google/object_detection/mobile_object_localizer_v1/1)

## Framework

TensorFlow Hub

## Model Format

ONNX opset=11

## Netron

[mobile_object_localizer.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/mobile_object_localizer/mobile_object_localizer.onnx.prototxt)
