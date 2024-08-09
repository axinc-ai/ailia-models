# TripoSR

## Input

![Input](input.png)

(Image from https://github.com/VAST-AI-Research/TripoSR/blob/main/examples/unicorn.png)

## Output

output is an .obj file

This video is rendered from an obj file

![Output](render.gif)


## Install

```
pip3 install -r requirements.txt
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,

``` bash
$ python3 TripoSR.py
```

If you want to specify the input point, put the .pts file path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.

```bash
$ python3 TripoSR.py --input POINT_FILE_PATH --savepath SAVE_IMAGE_PATH
```

## Reference

- [TripoSR](https://github.com/VAST-AI-Research/TripoSR)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

[TripoSR.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/TripoSR/TripoSR.onnx.prototxt)
[TripoSR_decoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/TripoSR/TripoSR_decoder.onnx.prototxt)
