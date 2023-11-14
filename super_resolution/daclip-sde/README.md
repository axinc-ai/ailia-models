# daclip-sde

## Input

![Input](input.jpg)


(Image from https://github.com/Algolzw/daclip-uir/tree/main/images/00006.jpg)

Ailia input shape : (1, 3, 256, 256)  

## Output

![Output](output.png)

Ailia output shape : (1, 3, 256, 256)

## Usage
Automatically downloads the onnx and prototxt files when running.
It is necessary to be connected to the Internet while downloading.

For the sample image with twice the resolution,
``` bash
$ python3 daclipsde.py
```

If you want to run in onnx mode, you specify --onnx option as below.

```bash
$ python3 daclipsde.py --onnx
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to chatge the name of the output file to save.
```bash
$ python3 daclipsde.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.

```bash
$ python3 daclipsde.py --video VIDEO_PATH
```

## Reference

[DA-CLIP](https://github.com/Algolzw/daclip-uir)

## Framework

Pytorch 2.1.0

## Model Format

ONNX opset = 17

## Netron

[daclip.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/daclip-sde/daclip.onnx.prototxt)

[universalIR.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/daclip-sde/universalIR.onnx.prototxt)
