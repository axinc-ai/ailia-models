# HANx2

Generates an image with twice the resolution.

## Input

![Input](000002_LR.png)

Ailia input shape : (1,3,194,194)  
Range : [0.0, 1.0]

## Output

![Output](output.png)

Ailia output shape : (1,3,388,388)  
Range : [0.0, 1.0]

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 han.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 han.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

## Reference

[Single Image Super-Resolution via a Holistic Attention Network](https://github.com/wwlCape/HAN.git)

## Framework

Pytorch 1.3.0

## Model Format

ONNX opset = 10

## Netron

[srresnet.onnx.prototxt](https://storage.googleapis.com/ailia-models/han/han_BIX2.onnx.prototxt)
https://storage.googleapis.com/ailia-models/han/han_BIX2.onnx
