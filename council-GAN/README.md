# Glasses Removal GAN

### Input
<img src='sample.jpg' width='320px'>


### Output
<img src='output.png' width='320px'>



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
$ python3 councilGAN-glasses.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 councilGAN-glasses.py --video VIDEO_PATH
```

### Reference
[Council-GAN](https://github.com/Onr/Council-GAN)

### Framework
PyTorch 1.5.1

### Model Format
ONNX opset = 10

### Netron

TO-DO