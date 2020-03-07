# Light Weight Human Pose Estimation 3D Demo

### input

![input_image](input.png)

Shape(1, 3,	360, 640) Range:[-1, 1]

### output

![output_image](ICV_3D_Human_Pose_Estimation_0.png)


### usage

For images:
``` bash
python3 lightweight-human-pose-estimation-3d.py --images input.png
```
argument `--rotate3d` to activate 3d-canvas-rotation-mode

For video:
```bash
python3 lightweight-human-pose-estimation-3d.py --video <VIDEO FILE>
```

In order to use web-camera,
```bash
python3 lightweight-human-pose-estimation-3d.py --video 0
```
(try -1 or 1 if 0 does not work, it may depend on your system)




### Reference

[Real-time 3D multi-person pose estimation demo in PyTorch. OpenVINO backend can be used for fast inference on CPU.](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation-3d-demo.pytorch)


### Framework
PyTorch 1.0


### Model Format
ONNX opset = 10


### Netron

[lightweight-human-pose-estimation-3d.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/lightweight-human-pose-estimation-3d/human-pose-estimation-3d.onnx.prototxt)
