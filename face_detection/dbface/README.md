# DBFace

## Input
<img src='selfie.png' width='480px'>

(Image from https://github.com/dlunion/DBFace/blob/master/datas/selfie.jpg)

## Output
<img src='selfie_output.png' width='480px'>

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
python3 dbface.py 
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 dbface.py  --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video. 
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 dbface.py --video VIDEO_PATH
```

## Reference
[DBFace : real-time, single-stage detector for face detection, with faster speed and higher accuracy](https://github.com/dlunion/DBFace)

## Framework
PyTorch

## Model Format
ONNX opset = 10

## Netron

[dbface_pytorch.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/dbface/dbface_pytorch.onnx.prototxt)