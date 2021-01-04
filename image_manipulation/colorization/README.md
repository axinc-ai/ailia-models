# Colorful Image Colorization

## Input

![Input](imgs/ansel_adams3.jpg)

(Image above is from [https://github.com/richzhang/colorization/tree/master/imgs](https://github.com/richzhang/colorization/tree/master/imgs))

## Output

![Output](imgs_out/ansel_adams3_output.jpg)

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```
$ python3 colorization.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```
$ python3 colorization.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```
$ python3 colorization.py --video VIDEO_PATH
```

## Reference

- [Colorful Image Colorization](https://github.com/richzhang/colorization)

## Framework

PyTorch

## Model Format

ONNX opset=10

## Netron

[colorizer.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/colorization/colorizer.onnx.prototxt)

