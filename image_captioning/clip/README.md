# CLIP

## Input

![Input](chelsea.png)

(Image from https://scikit-image.org/)

## Output

- Zero-Shot Prediction
```bash
### predicts the most likely top5 labels among input textual labels ###
    a cat: 98.40%
  a human: 1.35%
    a dog: 0.24%
```

## Requirements
This model requires additional module.

```
pip3 install ftfy
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 clip.py
```

If you want to specify the input image, put the image path after the `--input` option.
```bash
$ python3 clip.py --input IMAGE_PATH
```

You can use `--text` option  if you want to specify a subset of the texture labels to input into the model.  
Default labels is "a human", "a dog" and "a cat".
```bash
$ python3 clip.py --text "a human" --text "a dog" --text "a cat"
```

If you want to load a subset of the texture labels you input into the model from a file, use the --desc_file option.
```bash
$ python3 clip.py --desc_file imagenet_classes.txt
```

## Reference

- [CLIP](https://github.com/openai/CLIP)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[ViT-B32.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/clip/ViT-B32.onnx.prototxt)
