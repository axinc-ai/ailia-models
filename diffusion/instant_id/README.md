# InstantID

## Input
<img src="sample.jpg" width="256" height="256">

(Image from: https://github.com/InstantID/InstantID/blob/main/examples/yann-lecun_resize.jpg)

## Output
<img src="output_japanese.jpg" width="256" height="256">

(Prompt: Japanese anime style)

<img src="output_american.jpg" width="256" height="256">

(Prompt: American anime style)

## Usage
For the sample image, please execute command following with prompt text.

```bash
$ python3 instant_id.py -p/--prompt <prompt text>
```

If you want to infer using onnxruntime, set --onnx option.

```bash
$ python3 instant_id.py -p/--prompt <prompt text> --onnx
```

## Reference
[InstantID](https://github.com/InstantID/InstantID/tree/main)


## Framework
PyTorch


## Model Format
ONNX opset=11


## Netron
- [detection model](https://netron.app/?url=https://storage.googleapis.com/ailia-models/instant_id/detection.onnx.prototxt)
- [recognition model](https://netron.app/?url=https://storage.googleapis.com/ailia-models/instant_id/recognition.onnx.prototxt)
- [genderage model](https://netron.app/?url=https://storage.googleapis.com/ailia-models/instant_id/gender_age.onnx.prototxt)
- [landmark 2d 106 model](https://netron.app/?url=https://storage.googleapis.com/ailia-models/instant_id/landmark_2d_106.onnx.prototxt)
- [landmark 3d 68 model](https://netron.app/?url=https://storage.googleapis.com/ailia-models/instant_id/landmark_3d_68.onnx.prototxt)
