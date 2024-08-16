# YOLOP

## Input

![Input](input.jpg)

(Image from https://github.com/hustvl/YOLOP/blob/main/inference/images/0ace96c3-48481887.jpg)

Ailia input shape: (1, 3, 1280, 720)
Range: [-1.0, 1.0]

## Output

![Output](output.jpg)


## Usage
Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 yolop.py 
```

If you want to specify the input image, put the image path after the `--source` option.
You can use `--save-dir` option to change the name of the output file to save.
```bash
$ python3 yolop.py  --source inference/videos --save-dir inference/output
```


It can also be used for video files.
As with images, you can use them by specifying the path with the `--source` option.

```bash
$ python3 yolop.py --source inference/videos
```

## Reference

[YOLOP](https://github.com/hustvl/YOLOP)

## Model Format

ONNX opset = 11

## Framework

Pytorch

## Netron

[yolop.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/yolop/yolop.onnx.prototxt)

