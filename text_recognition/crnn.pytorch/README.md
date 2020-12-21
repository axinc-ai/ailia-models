# CRAFT: Character-Region Awareness For Text detection


## Input

![Input](demo.png)

(Image above is from [https://github.com/meijieru/crnn.pytorch](https://github.com/meijieru/crnn.pytorch).)

## Output

The output character will be printed.


## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```
$ python3 crnn.pytorch.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```
$ python3 craft-pytorch.py --input IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```
$ python3 crnn.pytorch.py --video VIDEO_PATH
```

## Reference

- [Convolutional Recurrent Neural Network](https://github.com/meijieru/crnn.pytorch)

## Framework

PyTorch

## Model Format

ONNX opset=10

## Netron

[crnn_pytorch.onnx.prototxt]()

