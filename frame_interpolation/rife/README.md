# Real-Time Intermediate Flow Estimation for Video Frame Interpolation

## Input

<img src="imgs/000001.png" width="240"><img src="imgs/000002.png" width="240">  

(Image from https://drive.google.com/file/d/1i3xlKb7ax7Y70khcTcuePi6E7crO_dFc/view?usp=sharing)

## Output

<img src="imgs_results/output_001.png" width="240">

## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample images,
```bash
$ python3 rife.py
```

If you want to specify the input image, put the first image path after the `--input` option, and the next image path after the `--input2` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 rife.py --input IMAGE_PATH1 --input2 IMAGE_PATH2 --savepath SAVE_IMAGE_PATH
```

The `--input` option can also specify the directory path where the images are located.
```bash
$ film rife.py --input DIR_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 rife.py --video VIDEO_PATH --savepath SAVE_VIDEO_PATH
```

for 4X interpolation.
```bash
$ python3 rife.py --exp 2
```

## Reference

- [ECCV2022-RIFE](https://github.com/megvii-research/ECCV2022-RIFE)

## Framework

Pytorch

## Model Format

ONNX opset=16

## Netron

[RIFE_HDv3.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/rife/RIFE_HDv3.onnx.prototxt)  
