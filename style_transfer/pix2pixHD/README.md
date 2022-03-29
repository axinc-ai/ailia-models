# pix2pixHD

## Input

### Input label map
<img src="frankfurt_000000_000576_gtFine_labelIds.png" width="512px">

### Input instance map
<img src="frankfurt_000000_000576_gtFine_instanceIds.png" width="512px">

(Images from https://www.cityscapes-dataset.com/)

## Output

<img src="output.png" width="512px">

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 pix2pixhd.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 pix2pixhd.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

## Reference

- [pix2pixHD](https://github.com/NVIDIA/pix2pixHD)

## Framework

PyTorch

## Model Format

ONNX opset=11

## Netron

[pix2pixhd.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/pix2pixhd/pix2pixhd.onnx.prototxt)
