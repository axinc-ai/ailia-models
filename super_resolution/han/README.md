# HAN

## Input

![Input](input.png)

(Image from https://github.com/sanghyun-son/EDSR-PyTorch/blob/master/test/0853x4.png)

Ailia input shape : (1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)  

## Output

![Output](output.png)

Ailia output shape : (1, 3, IMAGE_HEIGHT * scale, IMAGE_WIDTH * scale)

default : scale=2

## Usage
Automatically downloads the onnx and prototxt files when running.
It is necessary to be connected to the Internet while downloading.

For the sample image with twice the resolution (BI),
``` bash
$ python3 han.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 han.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

If you want to specify the scale for the resolution, put the scale after the `--scale` option.  
Choose the scale in [2, 3, 4, 8].
```bash
$ python3 han.py --scale SCALE 
```

If you want to the model trained on imaged degraded by the Blur-downscale Degradation Model (BD), specify the `--blur` option.  
Only a 3-resolution scale can be used with this option. 
```bash
$ python3 han.py --scale 3 --blur 
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.

```bash
$ python3 han.py --video VIDEO_PATH
```

The default setting is to use the optimized model and weights, but you can also switch to the normal model by using the `--normal` option.
If the output image is entirely black, try to add the `-e 0` option.
``` bash
$ python3 han.py -e 0
```

## Reference

[Single Image Super-Resolution via a Holistic Attention Network](https://github.com/wwlCape/HAN.git)

## Framework

Pytorch 1.3.0

## Model Format

ONNX opset = 11

## Netron

[han_BIX2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/han/han_BIX2.onnx.prototxt)
[han_BIX2.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/han/han_BIX2.opt.onnx.prototxt)

[han_BIX3.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/han/han_BIX3.onnx.prototxt)
[han_BIX3.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/han/han_BIX3.opt.onnx.prototxt)

[han_BIX4.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/han/han_BIX4.onnx.prototxt)
[han_BIX4.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/han/han_BIX4.opt.onnx.prototxt)

[han_BIX8.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/han/han_BIX8.onnx.prototxt)
[han_BIX8.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/han/han_BIX8.opt.onnx.prototxt)

[han_BDX3.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/han/han_BDX3.onnx.prototxt)
[han_BDX3.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/han/han_BDX3.opt.onnx.prototxt)
