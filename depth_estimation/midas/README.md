# MiDaS

## Input

![Input](input.jpg)

(Image from kitti dataset http://www.cvlibs.net/datasets/kitti/raw_data.php)

Shape : (1, 3, h, w)

## Output

![Output](input_depth.png)

Shape : (1, h, w)

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 midas.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 midas.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 midas.py --video VIDEO_PATH
```

By adding the `-v21` option, you can use version 2.1 model.  
(default use version 2.0 model)

If you use the version 2.1 model, you can use the small model with the `--model_type small` option.
```bash
$ python3 midas.py -v21 --model_type small
```

## Reference

[Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer](https://github.com/intel-isl/MiDaS)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[midas.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/midas/midas.onnx.prototxt)
[midas_v2.1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/midas/midas_v2.1.onnx.prototxt)
[midas_v2.1_small.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/midas/midas_v2.1_small.onnx.prototxt)
