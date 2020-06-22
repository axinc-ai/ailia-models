# Glasses Removal GAN

### Input
<img src='test.png' width='320px'>

Ailia input shape: (1, 128, 128, 1)  
Range: [0, 1]

### Output
![Result_image](output.png)



### usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 glasses_removal_gan.py 
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 glasses_removal_gan.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 glasses_removal_gan.py --video VIDEO_PATH
```

### Reference
[glasses-removal-gan](https://github.com/lecomte/glasses-removal-gan)

### Framework
Tensorflow 1.12.2

### Model Format
ONNX opset = 10

### Netron

[resnet_facial_feature.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/glasses_removal_gan/glasses_removal_gan.onnx.prototxt)
