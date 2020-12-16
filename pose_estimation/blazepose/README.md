# BlazePose

## Input

<!-- <img src="input.png" width="320px"> -->

<!-- (Image from https://github.com/hollance/BlazeFace-PyTorch/blob/master/3faces.png) -->

Ailia input shape: (1, 3, 256, 256)  
Range:

## Output


## Usage
<!-- Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 blazeface.py 
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 blazeface.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 blazeface.py --video VIDEO_PATH --savepath SAVE_VIDEO_PATH
``` -->

## Reference

[MediaPipePyTorch](https://github.com/zmurez/MediaPipePyTorch)

Squeeze function in Pytorch model removed as it produces an IF layer in ONNX unsupported in ailia (segmentation output has thus 1 more dim)

## Framework

PyTorch 1.7.1


## Model Format

ONNX opset = 11


<!-- ### Netron

[blazeface.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/blazeface/blazeface.onnx.prototxt) -->
