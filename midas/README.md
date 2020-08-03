# MiDaS

## Input

![Input](input.jpg)

Shape : (1, 3, 384, 384)  

## Output

![Output](input_depth.png)

Shape : (1, 384, 384)  

### usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 midas.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 midas.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 midas.py --video VIDEO_PATH
```

## Reference

## Framework

ONNX Runtime

## Model Format

ONNX opset=11

## Netron

[midas.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/mdias/midas.onnx.prototxt)
