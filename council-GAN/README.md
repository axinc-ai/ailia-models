# Council-GAN

### Input
<img src='sample.jpg' width='320px'>


### Output
<img src='output-glasses.png' width='320px'>

<img src='output-m2f.png' width='320px'>

<img src='output-anime.png' width='320px'>


### usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 council-gan.py --glasses
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 council-gan.py --glasses --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 council-gan.py --glasses --video VIDEO_PATH
```

### Reference
[Council-GAN](https://github.com/Onr/Council-GAN)

### Framework
PyTorch 1.5.1

### Model Format
ONNX opset = 10

### Netron

TO-DO