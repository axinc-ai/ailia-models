# ax_action_recognition

## Input

#### Video

![Video](punch_03-12-09-21-27-876.gif)

(Video is generated from dataset in https://github.com/felixchenfy/Realtime-Action-Recognition)

Input shape : (1, C, T, V, M) = (1, 2, 15, 18, 1)
```
C: channel number
T: frame number
V: keypoint number
M: person ID
```

## Output

![Output](punch_03-12-09-21-27-876_out.gif)

## Category

```
CATEGORY = (
'stand', 'walk', 'run', 'jump', 'sit', 'squat', 'kick', 'punch', 'wave'
)
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

By adding the `--video` option, It can be run as a real-time mode that infers frame by frame of the video.
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 ax_action_recognition.py --video VIDEO_PATH -a lw_human_pose
$ python3 ax_action_recognition.py --video VIDEO_PATH -a pose_resnet
```

## Reference

- [Realtime-Action-Recognition](https://github.com/felixchenfy/Realtime-Action-Recognition)
- [ST-GCN](https://github.com/yysijie/st-gcn)

The architecture of ax_action_recognition model is simplified from ST-GCN, and trained with Realtime-Action-Recognition dataset.

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[ax_action_recognition.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/ax_action_recognition/action.onnx.prototxt)
