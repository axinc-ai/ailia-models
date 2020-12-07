# pixel-link

## Input

![Input](img_249.jpg)

(Image from https://rrc.cvc.uab.es/?ch=4&com=downloads)

Shape : (height, width, 3)  

## Output

![Output](output.png)

- pixel_pos_scores shape : (1, 192, 320)
- link_pos_scores shape : (1, 192, 320, 8)

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 pixel_link.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 pixel_link.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 pixel_link.py --video VIDEO_PATH
```

## Reference

- [Pixel-Link](https://github.com/ZJULearning/pixel_link)

## Framework

Tesorflow

## Model Format

ONNX opset=11

## Netron

[pixellink-vgg16-4s.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/pixel_link/pixellink-vgg16-4s.onnx.prototxt)
