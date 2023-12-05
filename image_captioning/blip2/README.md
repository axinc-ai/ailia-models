# BLIP-2

## Input

![Input](merlion.png)

(Image from https://github.com/salesforce/LAVIS/blob/main/docs/_static/merlion.png)

## Output

- Estimating Caption
```bash
### Caption ###
singapore merlion fountain
```

## Requirements

This model requires additional module.

```
pip3 install transformers
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 blip2.py
```

If you want to specify the input image, put the image path after the `--input` option.  
```bash
$ python3 blip2.py --input IMAGE_PATH
```

## Reference

- [LAVIS - BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)
- [Hugging Face - BLIP-2](https://huggingface.co/spaces/Salesforce/BLIP2)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[blip2-opt-2.7b.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/blip2/blip2-opt-2.7b.onnx.prototxt)  
[vision_model.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/blip2/vision_model.onnx.prototxt)
