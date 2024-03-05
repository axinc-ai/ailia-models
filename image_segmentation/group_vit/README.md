# GroupViT: Semantic Segmentation Emerges from Text Supervision

## Input

![Input](voc.jpg)

(Image from https://github.com/NVlabs/GroupViT/blob/main/demo/examples/voc.jpg)

## Output

![Output](output.png)

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 group_vit.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 group_vit.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 group_vit.py --video VIDEO_PATH
```

You can specify the "model type" by specifying after the `--model_type` option.
The model type is selected from "yfcc", "redcap".  
```bash
$ python3 group_vit.py --model_type yfcc
```

To add an additional class label, specify it after the `--additional-class` option.
This can be specified with list of multiple items.
```bash
$ python3 group_vit.py --additional-class bookshelf
```

## Reference

- [GroupViT](https://github.com/NVlabs/GroupViT)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[group_vit_gcc_yfcc_30e-74d335e6.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/group_vit/group_vit_gcc_yfcc_30e-74d335e6.onnx.prototxt)  
[group_vit_gcc_yfcc_mlc.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/group_vit/group_vit_gcc_yfcc_mlc.onnx.prototxt)  
[group_vit_gcc_redcap_30e-3dd09a76.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/group_vit/group_vit_gcc_redcap_30e-3dd09a76.onnx.prototxt)  
[group_vit_gcc_redcap_mlc.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/group_vit/group_vit_gcc_redcap_mlc.onnx.prototxt)
