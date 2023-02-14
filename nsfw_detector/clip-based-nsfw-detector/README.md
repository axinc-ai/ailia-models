# CLIP-based-NSFW-Detector

## Input

Please download `_vyr_6097Sexy-Push-Up-Bikini-Brasilianisch-Bunt-2.jpg` from below url.

https://www.damenmode-quelle.de/fotky421/fotos/_vyr_6097Sexy-Push-Up-Bikini-Brasilianisch-Bunt-2.jpg

This image is used in the official demo.

https://colab.research.google.com/drive/19Acr4grlk5oQws7BHTqNIK-80XGw2u8Z?usp=sharing

## Output

- Estimating NSFW confidence
```bash
### Estimating NSFW confidence ###
 NSFW: 99.450
```

## Usage

Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet
while downloading.

For the sample image,
``` bash
$ python3 clip-based-nsfw-detector.py 
```

If you want to specify the input image, put the image path after the `--input` option.
```bash
$ python3 clip-based-nsfw-detector.py --input IMAGE_PATH
```

By adding the `--model_type` option, you can specify model type which is selected from "ViTB32", "ViTL14". (default is ViTB32)
```bash
$ python3 clip-based-nsfw-detector.py --model_type ViTB32
```

## Reference

- [CLIP-based-NSFW-Detector](https://github.com/LAION-AI/CLIP-based-NSFW-Detector)

## Framework

Keras

## Model Format

ONNX opset = 11

## Netron

[clip_bin_nsfw.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/clip-based-nsfw-detector/clip_bin_nsfw.onnx.prototxt)  
[clip_nsfw_b32.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/clip-based-nsfw-detector/clip_nsfw_b32.onnx.prototxt)
