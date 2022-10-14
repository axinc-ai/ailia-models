# 6D Rotation Representation for Unconstrained Head Pose Estimation (Pytorch)

## Input

![Input](input.png)

Image credit: "Day 21 Occupy Wall Street October 6 2011 Shankbone 6" by david_shankbone is marked under CC PDM 1.0. To view the terms, visit https://creativecommons.org/publicdomain/mark/1.0/

Ailia input shape: (1, 3, 224, 224)

## Output

![Output](output.png)

## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 6d_repnet.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 6d_repnet.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 6d_repnet.py --video VIDEO_PATH
```

The default setting is to use the optimized model and weights, but you can also switch to the normal model by using the --normal option.

## Reference

[6D Rotation Representation for Unconstrained Head Pose Estimation (Pytorch)](https://github.com/thohemp/6DRepNet)

## Framework

Pytorch

## Model Format

ONNX opset = 11

## Netron

[6DRepNet.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/6d_repnet/6DRepNet.opt.onnx.prototxt)

[RetinaFace.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/6d_repnet/RetinaFace.opt.onnx.prototxt)
