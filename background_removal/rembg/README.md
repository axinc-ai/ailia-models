# Rembg

## Input

![Input](animal-1.jpg)

(Image from https://github.com/danielgatis/rembg/blob/main/examples/animal-1.jpg)

## Output

![Output](output.png)

## Requirements
This model requires additional module.

```
pip3 install pymatting
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 rembg.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 rembg.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

Add the `--composite` option if you want to combine the input image with the calculated alpha value.
```bash
$ python3 rembg.py --composite
```

## Reference

- [Rembg](https://github.com/danielgatis/rembg)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[u2net_opset11.onnx.prototxt](https://storage.googleapis.com/ailia-models/u2net/u2net_opset11.onnx.prototxt)
