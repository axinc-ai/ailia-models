# hat

## Input

![Input](input.png)


(Image from https://github.com/JingyunLiang/SwinIR/tree/main/testsets)

Ailia input shape : (1, 3, 256, 256)  

## Output

![Output](output.png)

Ailia output shape : (1, 3, 256 * scale, 256 * scale)

default : scale=2

## Usage
Automatically downloads the onnx and prototxt files when running.
It is necessary to be connected to the Internet while downloading.

For the sample image with twice the resolution (BI),
``` bash
$ python3 hat.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to chatge the name of the output file to save.
```bash
$ python3 hat.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

["HAT","HAT_S","HAT_GAN_REAL_sharper","HAT_GAN_REAL"],

By adding the `--arch` option, you can specify model type which is selected from "HAT","HAT_S","HAT_GAN_REAL_sharper","HAT_GAN_REAL".  
(default is HAT)
```bash
$ python3 hat.py --arch HAT
```


If you want to specify the scale for the resolution, put the scale after the `--scale` option.  
Choose the scale in [2, 3, 4].
```bash
$ python3 hat.py --scale SCALE 
```


By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.

```bash
$ python3 hat.py --video VIDEO_PATH
```

arch 説明

## Reference

[Hat](https://github.com/XPixelGroup/HAT)

## Framework

Pytorch 1.10.0

## Model Format

ONNX opset = 11

## Netron

[Hat_S_x2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/hat/Hat_S_x2.onnx.prototxt)

[Hat_S_x3.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/hat/Hat_S_x3.onnx.prototxt)

[Hat_S_x4.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/hat/Hat_S_x4.onnx.prototxt)

[Hat_x2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/hat/Hat_x2.onnx.prototxt)

[Hat_x3.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/hat/Hat_x3.onnx.prototxt)

[Hat_x4.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/hat/Hat_x4.onnx.prototxt)

[HAT_GAN_REAL.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/hat/HAT_GAN_REAL.onnx.prototxt)

[HAT_GAN_REAL_sharper.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/hat/HAT_GAN_REAL_sharper.onnx.prototxt)
