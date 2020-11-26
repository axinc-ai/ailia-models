# U^2-Net

### input
![input_image](input.png)  
(Image from https://github.com/NathanUA/U-2-Net/blob/master/test_data/test_images/girl.png)
- Ailia input shape: (1, 3, 320, 320)  

### output
![output_image](output.png)

### usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 u2net.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 u2net.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 u2net.py --video VIDEO_PATH
```

You can select a pretrained model by specifying `-a large`(default) or `-a small`.

```bash
$ python3 u2net.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH -a small
```

When using ailia SDK 1.2.4 or later, you can use a highly accurate model by specifying `--opset 11`.

```bash
$ python3 u2net.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH --opset 11
```

Add the `--composite` option if you want to combine the input image with the calculated alpha value.

```bash
$ python3 u2net.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH --opset 11 --composite
```

### Reference

[U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://github.com/NathanUA/U-2-Net)


### Framework
PyTorch 1.1


### Model Format
ONNX opset = 10


### Netron

[u2net.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/u2net/u2net.onnx.prototxt)

[u2netp.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/u2net/u2netp.onnx.prototxt)

[u2net_opset11.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/u2net/u2net_opset11.onnx.prototxt)

[u2netp_opset11.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/u2net/u2netp_opset11.onnx.prototxt)
