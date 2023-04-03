# PyTorch-FCN

## Input
![Input](image.jpg)

(Image above are from []())

## Output

![Output](result.jpg)

## Usae

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```
$ python3 pytorch-fcn.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```
$ python3 pytorch-fcn --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

## Reference

- [pytorch-fcn](https://github.com/wkentaro/pytorch-fcn)

## Framework

PyTorch

## Model Format

ONNX opset=11

## Netron

[pytorch-unet.onnx.prototxt](https://storage.googleapis.com/ailia-models/pytorch_fcn/pytorch_fcn32s.onnx.prototxt)
