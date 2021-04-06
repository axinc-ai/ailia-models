# GAST-Net-3DPoseEstimation

## Input

<img src="img/baseball_040.png" width="240"><img src="img/baseball_050.png" width="240"><img src="img/baseball_060.png" width="240">  

(Video from https://github.com/fabro66/GAST-Net-3DPoseEstimation/blob/master/data/video/baseball.mp4)

## Output

<img src="img/output_040.png" width="500"><img src="img/output_050.png" width="500"><img src="img/output_060.png" width="500">  

## Requirements

This model requires additional module and specific version of module.

```
pip3 matplotlib==3.1.0
pip3 install filterpy
```

And following dependencies is required for creating .mp4 file.

- ffmpeg

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample video,
```bash
$ python3 gast.py
```

If you want to specify the input video, put the video file path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save. In addition to .mp4, .gif can be specified as the extension of the save file.
```bash
$ python3 gast.py --input VIDEO_PATH --savepath SAVE_VIDEO_PATH
```

The default mode is Single-person 3D pose estimation.  
For Two-person 3D pose estimation, you can specify `-np 2` option.
```bash
$ python3 gast.py -np 2
```

The Gast-Net adopt YOLOv3 for human detection.  
It use implementation of ailia for YOLOv3.
You can use `-dn` option to use implementation of Pytorch. (Then pytorch module is required.)
```bash
$ python3 gast.py -dn
```

## Reference

- [A Graph Attention Spatio-temporal Convolutional Networks for 3D Human Pose Estimation in Video (GAST-Net)](https://github.com/fabro66/GAST-Net-3DPoseEstimation) 
- [YOLOv3](https://github.com/ayooshkathuria/pytorch-yolo-v3) 
- [SORT](https://github.com/abewley/sort) 
- [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) 
- [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) 

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[27_frame_17_joint_model.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gast/27_frame_17_joint_model.onnx.prototxt)  
[pose_hrnet_w48_384x288.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/gast/pose_hrnet_w48_384x288.onnx.prototxt)  
[yolov3.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolov3/yolov3.opt.onnx.prototxt)  
