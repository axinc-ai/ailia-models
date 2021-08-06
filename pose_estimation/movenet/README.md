# MoveNet

## Input

<img src="input.jpg" width="320px">

(Image from https://images.pexels.com/photos/4384679/pexels-photo-4384679.jpeg)

Model variant: Thunder  
Ailia input shape : (1, 256, 256, 3)  
Range : [0, 1.0]

Model variant: Lightning  
Ailia input shape : (1, 192, 192, 3)  
Range : [0, 1.0]

## Output

<img src="output.png" width="320px">

- Confidence : (1, 1, 17, 3)
- Range : [0, 1.0]

## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 movenet.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 movenet.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

If you want to specify the model variant, put the model variant after the `--model_variant` option.  
You can only choose variants from 'thunder','lightning'. default is 'thunder'
```bash
$ python3 movenet.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH --model_variant lightning
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 movenet.py --video VIDEO_PATH
```

## Reference

[Code repo for movenet](https://www.tensorflow.org/hub/tutorials/movenet)

## Framework

TensorFlow

## Model Format

ONNX opset = 11

## Netron
[movenet_thunder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/movenet/movenet_thunder.onnx.prototxt)

[movenet_lightning.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/movenet/movenet_lightning.onnx.prototxt)