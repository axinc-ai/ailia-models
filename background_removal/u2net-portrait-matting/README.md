# U^2-Net - Portrait matting

### Input
![input_image](input.png)  
(Image from https://github.com/dennisbappert/u-2-net-portrait/blob/master/docs/sample_1.jpeg)
- Ailia input shape: (1, 3, 448, 448)  

### Output
![output_image](output.png)

### Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 u2net-portrait-matting.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 u2net-portrait-matting.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 u2net-portrait-matting.py --video VIDEO_PATH
```

Add the `--composite` option if you want to combine the input image with the calculated alpha value.
```bash
$ python3 u2net-portrait-matting.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH --composite
```

### Reference

- [U^2-Net - Portrait matting](https://github.com/dennisbappert/u-2-net-portrait)
- [U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://github.com/NathanUA/U-2-Net)

### Framework

PyTorch

### Model Format

ONNX opset = 11

### Netron

[u2net-portrait-matting.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/u2net-portrait-matting/u2net-portrait-matting.onnx.prototxt)
