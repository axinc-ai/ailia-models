# NeRF

## Input
![Input](./sample.png)

(from https://drive.google.com/file/d/1P6KIDAr68twCJqcq2BJH6cgTOXXxAFIv/view?usp=sharing)

input shape: (3, 756, 1008)

## Output
###LLFF Fern video
![Output](./data/nerf_llff_data/output/sample.gif)


## Note

This Software is licensed under the Attribution-NonCommercial-ShareAlike 4.0 International.

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
python3 nerf.py 
```

## Reference
[NeRF: Neural Radiance Fields](https://github.com/bmild/nerf)

## Framework
Tensorflow 1.15

## Model Format
ONNX opset = 12

## Netron

[nerf.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/nerf/nerf.opt.onnx.prototxt)