# SKU110K-DenseDet

## Input

![Input](demo.jpg)

(Image from https://github.com/eg4000/SKU110K_CVPR19)

Shape : (1, 3, 2560, 2560)

## Output

![Output](output.png)

- det_bboxes shape : (n, 5)
- det_labels shape : (n,)

## Usage

Automatically downloads the onnx and prototxt files on the first run. It is necessary to be connected to the Internet
while downloading.

For the sample image,

```bash
$ python3 sku110k-densedet.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.

```bash
$ python3 sku110k-densedet.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.

```bash
$ python3 sku110k-densedet.py --video VIDEO_PATH
```

## Reference

- [SKU110K-DenseDet](https://github.com/Media-Smart/SKU110K-DenseDet)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[DenseDet.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/sku110k-densedet/DenseDet.onnx.prototxt)
