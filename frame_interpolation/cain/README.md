# CAIN

## Input

<img src="sample/im3.png" width="240"><img src="sample/im5.png" width="240">  

(Image from Vimeo-90K dataset http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip)

## Output

<img src="sample_results/output_0.png" width="240">

## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample images,
```bash
$ python3 cain.py
```

If you want to specify the input image, put the first image path after the `--input` option, and the next image after the `--input2` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 cain.py --input IMAGE_PATH1 --input2 IMAGE_PATH2 --savepath SAVE_IMAGE_PATH
```

The `--input` option can also specify the directory path where the images are located.
```bash
$ python3 cain.py --input DIR_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 cain.py --video VIDEO_PATH --savepath SAVE_VIDEO_PATH
```

You can set output video size using `-hw` option.
```bash
$ python3 cain.py -hw 256,448
```

## Reference

- [Channel Attention Is All You Need for Video Frame Interpolation](https://github.com/myungsub/CAIN)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[cain.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/cain/cain.onnx.prototxt)  
