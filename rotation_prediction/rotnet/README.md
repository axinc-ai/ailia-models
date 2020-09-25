# RotNet

### input
<img src='test.jpg' width=300px>

(from https://github.com/d4nst/RotNet/tree/master/data/test_examples)

Ailia input shape: (1, 224, 224, 3)

### output
- Original: original image (after cropped)
- Rotated: input image (randomly rotated)
- Corrected: output image (model output is predicted angle, therefore we rotated the "rotated image" to visualize our output)
![output_image](output.png)


### Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 rotnet.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 rotnet.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 rotnet.py --video VIDEO_PATH
```

Currectly, two pretrained-models are avilable:
- mnist(for mnist dataset)
- gsv2(for google street view dataset)
You can select one of them by adding `--model` (default: gsv2).


### Reference
[CNNs for predicting the rotation angle of an image to correct its orientation](https://github.com/d4nst/RotNet)

### Framework
Keras

### Model Format
ONNX opset = 10

### Netron

[rotnet_gsv_2.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/rotnet/rotnet_gsv_2.onnx.prototxt)
