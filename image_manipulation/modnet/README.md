# MODNet

## Input

![Input](input.jpg)
Ailia input shape: (1, 3, 512, 736)

## Output

![Output](output.jpg)

## Usage

Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet
while downloading.

For the sample image,

``` bash
$ python3 modnet.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.

```bash
$ python3 modnet.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.

```bash
$ python3 modnet.py --video VIDEO_PATH
```

## Reference

[modnet](https://github.com/ZHKKKe/MODNet)

## Framework

Pytorch

## Model Format

ONNX opset = 11

## Netron

[modnet.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/modnet/modnet.opt.onnx.prototxt)
