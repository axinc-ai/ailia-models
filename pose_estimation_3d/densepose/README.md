# Towards 3D Human Pose Estimation in the Wild: a Weakly-supervised Approach

## Input

![Input](input.jpg)

(Image from https://cocodataset.org/)

Ailia input shape: (1, 3, 640, 427)

## Output

![Output](output.png)


## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 densepose.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 densepose.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 densepose.py --video VIDEO_PATH
```

The default setting is to use the optimized model and weights, but you can also switch to the normal model by using the --normal option.

## Reference

[DensePose in Detectron2
](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose)

## Framework

Pytorch 1.13.0

## Model Format

ONNX opset = 14

## Netron


[R_101_FPN_s1x.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/densepose/R_101_FPN_s1x.onnx.prototxt)


[R_101_FPN_DL_WC1M_six.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/densepose/R_101_FPN_DL_WC1M_six.onnx.prototxt)

[R_101_FPN_DL_WC1_s1x.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/densepose/R_101_FPN_DL_WC1_s1x.onnx.prototxt)

[R_101_FPN_DL_WC2M_six.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/densepose/R_101_FPN_DL_WC2M_six.onnx.prototxt)

[R_101_FPN_DL_WC2_s1x.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/densepose/R_101_FPN_DL_WC2_s1x.onnx.prototxt)

[R_101_FPN_DL_s1x.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/densepose/R_101_FPN_DL_s1x.onnx.prototxt)

[R_101_FPN_WC1M_six.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/densepose/R_101_FPN_WC1M_six.onnx.prototxt)

[R_50_FPN_DL_WC1M_six.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/densepose/R_50_FPN_DL_WC1M_six.onnx.prototxt)

[R_50_FPN_DL_WC1_s1x.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/densepose/R_50_FPN_DL_WC1_s1x.onnx.prototxt)

[R_50_FPN_DL_WC2M_six.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/densepose/R_50_FPN_DL_WC2M_six.onnx.prototxt)

[R_50_FPN_DL_WC2_s1x.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/densepose/R_50_FPN_DL_WC2_s1x.onnx.prototxt)

[R_50_FPN_DL_s1x.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/densepose/R_50_FPN_DL_s1x.onnx.prototxt)

[R_50_FPN_WC1M_six.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/densepose/R_50_FPN_WC1M_six.onnx.prototxt)

[R_50_FPN_WC1_s1x.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/densepose/R_50_FPN_WC1_s1x.onnx.prototxt)

[R_50_FPN_WC2M_six.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/densepose/R_50_FPN_WC2M_six.onnx.prototxt)

[R_50_FPN_WC2_s1x.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/densepose/R_50_FPN_WC2_s1x.onnx.prototxt)

[R_50_FPN_s1x.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/densepose/R_50_FPN_s1x.onnx.prototxt)
