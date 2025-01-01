# SberSwap

## Input

<table>
<tr>
<th>Source image</th>
<th>Target image</th>
</tr>
<tr>
<td><img src="elon_musk.jpg" width="320px"></td>
<td><img src="beckham.jpg" width="320px"></td>
</tr>
</table>

(Image from https://github.com/ai-forever/sber-swap/tree/main/examples/images)

## Output

<img src="output.png" width="320px">

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 sber-swap.py
```

If you want to specify the target image, put the image path after the `--input` option.  
The source image can be specified with the `--source` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 sber-swap.py --input TARGET_IMAGE --source SOURCE_IMAGE --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 sber-swap.py --video VIDEO_PATH
```

## Reference

- [SberSwap](https://github.com/ai-forever/sber-swap)

## Framework

Pytorch

## Model Format

- G_unet_2blocks.onnx.prototxt
- scrfd_10g_bnkps.onnx.prototxt
- arcface_backbone.onnx.prototxt

  ONNX opset=11


- face_landmarks.onnx.prototxt

  ONNX opset=12

## Netron

[G_unet_2blocks.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/sber-swap/G_unet_2blocks.onnx.prototxt)  
[scrfd_10g_bnkps.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/sber-swap/scrfd_10g_bnkps.onnx.prototxt)  
[arcface_backbone.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/sber-swap/arcface_backbone.onnx.prototxt)  
[face_landmarks.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/sber-swap/face_landmarks.onnx.prototxt)
