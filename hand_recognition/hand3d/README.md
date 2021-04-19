# ColorHandPose3D

## Input

![Input](img.png)

(Image from https://lmb.informatik.uni-freiburg.de/projects/hand3d/ColorHandPose3D_data_v3.zip)

Shape : (1, 240, 320, 3)

## Output

![Output](output.png)

Shape : (1, 240, 320, 2)

### Usage

Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet
while downloading.

For the sample image,

``` bash
$ python3 hand3d.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.

```bash
$ python3 hand3d.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.

```bash
$ python3 hand3d.py --video VIDEO_PATH
```

## Reference

[ColorHandPose3D network](https://github.com/lmb-freiburg/hand3d)

## Framework

TensorFlow

## Model Format

ONNX opset=11

## Netron

[hand_scoremap.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/hand3d/hand_scoremap.onnx.prototxt)
