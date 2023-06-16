# High-Speed face Emotion recognition

## Input

![Input](lenna.png)

- Ailia input shape: (1, 3, 224, 224) RGB channel order
  - Ailia input shape: (1, 3, 260, 260) for B2 model
- Pixel value range: [0, 1] before normalization
- Preprocessing: normalization using ImageNet statistics

## Output

```
emotion_class_count=4
+ idx=0
  category=5 [ Neutral ]
  prob=0.6248039603233337
+ idx=1
  category=4 [ Happiness ]
  prob=0.15010859072208405
+ idx=2
  category=7 [ Surprise ]
  prob=0.07648341357707977
+ idx=3
  category=6 [ Sadness ]
  prob=0.05946649610996246
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 hsemotion.py 
```

If you want to specify the input image, put the image path after the `--input` option.  
```bash
$ python3 hsemotion.py --input IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 hsemotion.py --video VIDEO_PATH
```


## Reference

[High-Speed face Emotion recognition](https://github.com/HSE-asavchenko/face-emotion-recognition)

## Framework

PyTorch

## Model Format

ONNX opset = 11

## Netron

- [enet_b0_8_best_afew.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/hsemotion/enet_b0_8_best_afew.onnx.prototxt)
- [enet_b0_8_best_vgaf.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/hsemotion/enet_b0_8_best_vgaf.onnx.prototxt)
- [enet_b0_8_va_mtl.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/hsemotion/enet_b0_8_va_mtl.onnx.prototxt)
- [enet_b2_8.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/hsemotion/enet_b2_8.onnx.prototxt)