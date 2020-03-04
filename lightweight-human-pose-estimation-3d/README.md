# BlazeFace

### input

![input_image](input.png)

Shape(1, 3,	256, 448) Range:[-1, 1]

### output

![output_image](ICV_3D_Human_Pose_Estimation_0.png)


### usage

For images:
``` bash
python3 lightweight-human-pose-estimation-3d.py --images input.png
```
argument `ratate3d` to activate 3d-canvas-rotation-mode

In order to use web-camera,
```bash
python3 lightweight-human-pose-estimation-3d.py --video 0
```
(try -1 or 1 if 0 does not work, it may depends on your system)





### Reference

[BlazeFace-PyTorch](https://github.com/hollance/BlazeFace-PyTorch)


### Framework
PyTorch 1.1


### Model Format
ONNX opset = 10
