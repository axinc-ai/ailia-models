# lightweight-human-pose-estimation

## Input

![Input](balloon.png)

Shape : (1, 3, 240, 320)
Range : [-0.5, 0.5]

## Output

![Output](output.png)

- Confidence : (1, 19, 30, 40)
- Range : [0, 1.0]

## Usage

Predict from image

```
python3 lightweight-human-pose-estimation.py
```

Predict from web camera

```
python3 lightweight-human-pose-estimation.py video
```

## Reference

[Fast and accurate human pose estimation in PyTorch. Contains implementation of "Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose" paper.](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch)

## Framework

Pytorch 1.2.0

## Model Format

ONNX opset = 10

## Netron

[lightweight-human-pose-estimation.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/lightweight-human-pose-estimation/lightweight-human-pose-estimation.onnx.prototxt)

[lightweight-human-pose-estimation.opt.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/lightweight-human-pose-estimation/lightweight-human-pose-estimation.opt.onnx.prototxt)

