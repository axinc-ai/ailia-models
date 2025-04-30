# SAMURAI

## Input
- A video file or a directory containing images
- The bounding box of the object in the first frame

## Output
A video file with the segmented object across all frames.

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample video,
```bash
$ python3 samurai.py
```

If you want to specify the video image, put the file path after the `--video` option.  
You can use `--savepath` option to specify the name of the output file to save.
```bash
$ python3 samurai.py --video VIDEO_PATH --savepath SAVE_FILE_PATH
```

To provide bounding box information for the first frame,
specify a .txt file using the `--txt_path` option.
```bash
$ python3 samurai.py --txt_path TXT_PATH
```
Note: The .txt file should contain a single line with the bounding box in x,y,w,h format (top-left coordinates, width, and height).

## Reference

- [SAMURAI](https://github.com/yangchris11/samurai)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[backbone.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/samurai/backbone.onnx.prototxt)  
[sam_heads.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/samurai/sam_heads.onnx.prototxt)  
[memory_encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/samurai/memory_encoder.onnx.prototxt)  
[memory_attention.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/samurai/memory_attention.onnx.prototxt)  
