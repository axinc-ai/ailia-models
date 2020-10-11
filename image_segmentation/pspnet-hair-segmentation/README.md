# psp-net Hair Segmentation

### input
![input_image](test.jpg)

(from http://projects.i-ctm.eu/it/progetto/figaro-1k)

Ailia input shape: (1, 3, 512, 512)  

### output
![output_image](output.png)

### usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 pspnet-hair-segmentation.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 pspnet-hair-segmentation.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 pspnet-hair-segmentation.py --video VIDEO_PATH
```

### Reference

[pytorch-hair-segmentation](https://github.com/YBIGTA/pytorch-hair-segmentation)


### Framework
PyTorch 0.4.1


### Model Format
ONNX opset = 10


### Netron
[pspnet-hair-segmentation.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/pspnet-hair-segmentation/pspnet-hair-segmentation.onnx.prototxt)
