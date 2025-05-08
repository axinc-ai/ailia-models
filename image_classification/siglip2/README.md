# siglip

## Input

![Input](test.jpg)

(Image from https://farm9.staticflickr.com/8225/8511402100_fea15da1c5_z.jpg)

## Output

- Zero-Shot Prediction
```bash
1: tiger cat - 24.91%
2: tabby, tabby cat - 13.09%
3: Egyptian cat - 12.17%
4: computer keyboard, keypad - 6.68%
5: remote control, remote - 2.12%
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 siglip.py
```

If you want to specify the input image, put the image path after the `--input` option.
```bash
$ python3 siglip.py --input IMAGE_PATH
```

## Reference

- [Zero-shot Image Classification with SigLIP2](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/siglip-zero-shot-image-classification/siglip-zero-shot-image-classification.ipynb)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

[regnet_y_800mf.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/siglip/regnet_y_800mf.onnx.prototxt)  
