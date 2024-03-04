# Japanese Stable CLIP ViT-L/16

## Input

![Input](dog.jpeg)

(Image from https://images.pexels.com/photos/2253275/pexels-photo-2253275.jpeg)

## Output

```bash
class_count=3
+ idx=0
  category=0[犬 ]
  prob=1.0
+ idx=1
  category=2[象 ]
  prob=0.0
+ idx=2
  category=1[猫 ]
  prob=0.0
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
$ python3 japanese-stable-clip-vit-l-16.py
```

If you want to specify the input image, put the image path after the `--input` option.
```bash
$ python3 japanese-stable-clip-vit-l-16.py --input IMAGE_PATH
```

You can use `--text` option  if you want to specify a subset of the texture labels to input into the model.  
Default labels is "犬", "猫" and "象".
```bash
$ python3 japanese-stable-clip-vit-l-16.py --text "犬" --text "猫" --text "象"
```

## Reference

- [Hugging Face - Japanese Stable CLIP ViT-L/16](https://huggingface.co/stabilityai/japanese-stable-clip-vit-l-16)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[CLIP-ViT-L16-image.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/japanese-stable-clip-vit-l-16/CLIP-ViT-L16-image.onnx.prototxt)  
[CLIP-ViT-L16-text.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/japanese-stable-clip-vit-l-16/CLIP-ViT-L16-text.onnx.prototxt)  
