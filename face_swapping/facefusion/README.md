# Facefusion

## Inputs

**Source image**

[<img src="source.jpg" width=512px>](source.jpg)

(Image from https://github.com/facefusion/facefusion-assets/releases/download/examples/source.jpg)

**Target image**

[<img src="target.jpg" width=512px>](target.jpg)

(Image from https://github.com/facefusion/facefusion-assets/releases/download/examples/target-1080p.mp4)

## Output

[<img src="output.png" width=512px>](output.png)

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 facefusion.py
```

If you want to specify the input image, put the image path after the `--input` option.
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 facefusion.py --input TARGET_IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

If you want to specify the source image, put the image path after the `--source` option.
```bash
$ python3 facefusion.py --input TARGET_IMAGE_PATH --source SOURCE_IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 facefusion.py --video VIDEO_PATH
```

## Reference

- [Facefusion](https://github.com/facefusion/facefusion)

## Framework

ONNXRuntime

## Model Format

ONNX opset=11 for `2dfan4.onnx`, `arcface_w600k_r50.onnx`, `inswapper_128.onnx`

ONNX opset=12 for `yoloface_8n.onnx`

## Netron

- [yoloface_8n.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/facefusion/yoloface_8n.onnx.prototxt)
- [2dfan4.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/facefusion/2dfan4.onnx.prototxt)
- [arcface_w600k_r50.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/facefusion/arcface_w600k_r50.onnx.prototxt)
- [inswapper_128.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/facefusion/inswapper_128.onnx.prototxt)
