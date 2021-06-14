# Robust Neural Network Output Enhancement for Active Lane Detection (RONELD)

## Input

![Input](input.jpg)

(Image from https://github.com/czming/RONELD-Lane-Detection/tree/main/example/00000.jpg)

Ailia input shape: (1, 3, 208, 976)

## Output

![Output](output.jpg)

## Usage

Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet
while downloading.

For the sample image,

``` bash
$ python3 roneld.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.

```bash
$ python3 roneld.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.

```bash
$ python3 roneld.py --video VIDEO_PATH
```

The default setting is to use the optimized model and weights, but you can also switch to the normal model by using the
--normal option.

## Reference

[RONELD-Lane-Detection](https://github.com/czming/RONELD-Lane-Detection)

## Framework

Pytorch

## Model Format

ONNX opset = 11

## Netron

[erfnet.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/erfnet/erfnet.opt.onnx.prototxt)