# ReStyle

## Input

[<img src="img/face_img.jpg" width=256px>](img/face_img.jpg)
[<img src="img/toonify_img.jpg" width=256px>](img/toonify_img.jpg)

(Image from https://github.com/yuval-alaluf/restyle-encoder/blob/main/notebooks/images/)

Shape : (1, 3, 1024, 1024)

Face alignment and reshaped to : (1, 3, 256, 256)  

## Output

### Encoder

![Output](img/output.png)

### Toonification

![Toonified Output](img/output_toonify.png)

\* Note: From left to right: 1st, 2nd, 3rd, 4th, 5th iteration, and original (face aligned) image.

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image (running on CPU),
```bash
$ python3 restyle-encoder.py -e 0
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 restyle-encoder.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH -e 0
```

By specifying the `-iter` option, you can choose how many iterations you want to generate the output image (default 5).
```bash
$ python3 restyle-encoder.py -iter 3 -e 0
```

By specifying the `-toon` option, you can run the toonification task.
```bash
$ python3 restyle-encoder.py -toon --input img/toonify_img.jpg --savepath img/output_toonify.png -e 0
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 restyle-encoder.py --video VIDEO_PATH -e 0
```

## Reference

- [ReStyle](https://github.com/yuval-alaluf/restyle-encoder)

## Framework

Pytorch 1.10.0

Python 3.6.7+

## Model Format

ONNX opset=11

## Netron

[restyle-encoder.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/restyle-encoder/restyle-encoder.onnx.prototxt)

[face-pool.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/restyle-encoder/face-pool.onnx.prototxt)

[toonify.onnx.prototxt](https://lutzroeder.github.io/netron/?url=https://storage.googleapis.com/ailia-models/restyle-encoder/toonify.onnx.prototxt)
