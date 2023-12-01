# LightGlue: Local Feature Matching at Light Speed

## Input

- img_A

![Input](img_A.png)

- img_B

![Input](img_B.png)

(Image from https://github.com/ufukefe/DFM/tree/main/python/data)

## Output

- Matches result

![Output](output.png)

## Caution
The software can only be used for personal, research, academic and non-commercial purposes.

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 lightglue.py
```

If you want to specify the input image, put the image path (as img_B) after the `--input` option, 
and the second image path (as img_A) after the `--input2` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 lightglue.py --input IMAGE_B --input2 IMAGE_A --savepath SAVE_IMAGE_PATH
```

## Reference

- [LightGlue](https://github.com/fabio-sim/LightGlue-ONNX)

## Framework

Pytorch

## Model Format

ONNX opset=16

## Netron

[lightglue.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/lightglue/lightglue.onnx.prototxt)

[superpoint_lightglue.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/lightglue/superpoint_lightglue.onnx.prototxt)
