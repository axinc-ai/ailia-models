# PolyLaneNet: Lane Estimation via Deep Polynomial Regression

### input
Images with the same aspect ratio as 360×640. This model detect lane from inputs.
- input image
![入力画像](./raws/original.jpg)

### output
Images which the detected lane in input images is colored in green.
- output image (pverlayed)
![出力画像1](./raws/overlayed.jpg)

- output image (predicted)
![出力画像2](./raws/predicted.jpg)

### Usage
```bash
$python3 polylanenet.py
```


## Reference

[PolyLaneNet](https://github.com/lucastabelini/PolyLaneNet)

## Framework

PyTorch 1.9.0


## Model Format

ONNX opset = 11

## Netron

[polylanenet.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/polylanenet/polylanenet.onnx.prototxt)
