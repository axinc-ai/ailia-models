# BlazeFace

### input

<img src="input.png" width="320px">

(Image from https://search.creativecommons.org/photos/df3a19c2-47ca-4f58-8aed-0dc62e89e9e9)

Image credit: "Day 21 Occupy Wall Street October 6 2011 Shankbone 6" by david_shankbone is marked under CC PDM 1.0. To view the terms, visit https://creativecommons.org/publicdomain/mark/1.0/


Ailia input shape: (1, 3, 128, 128)  
Range: [-1, 1]

### output
![output_image](result.png)

### usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 blazeface.py 
```

If you want to use the blazefaceback model, add the `--back` or `-b` option.   
You can find details in the reference.   
To summarize, blazefaceback is the model that is trained to match the back-facing camera.
```bash
$ python3 blazeface.py --back
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 blazeface.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 blazeface.py --video VIDEO_PATH --savepath SAVE_VIDEO_PATH
```

### Reference

[BlazeFace-PyTorch](https://github.com/hollance/BlazeFace-PyTorch)


### Framework
PyTorch 1.1


### Model Format
ONNX opset = 10


### Netron

[blazeface.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/blazeface/blazeface.onnx.prototxt)
