# Vision Transformer

## input
![input image](input.jpg)

(from https://pixabay.com/ja/photos/%E5%BA%97-%E3%82%B9%E3%83%BC%E3%83%91%E3%83%BC%E3%83%9E%E3%83%BC%E3%82%B1%E3%83%83%E3%83%88-%E3%82%B9%E3%82%AB%E3%83%BC%E3%83%88-4527402/)

<br/>

## output
![output_image](output.png)

<br/>

## usage
Automatically downloads the onnx and prototxt files on the first run.  
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python mivolo.py --onnx
(ex on CPU)  $ python mivolo.py --onnx -e 0
(ex on BLAS) $ python mivolo.py --onnx -e 1
(ex on GPU)  $ python mivolo.py --onnx -e 2
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 mivolo.py --onnx --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
$ python3 mivolo.py --onnx -i IMAGE_PATH -s SAVE_IMAGE_PATH
(ex) $ python3 mivolo.py --onnx --input input.jpg --savepath output.png
```

By adding the `--video` option, you can input the video.
```bash
$ python3 mivolo.py --onnx --video VIDEO_PATH --savepath SAVE_VIDEO_PATH
$ python3 mivolo.py --onnx -v VIDEO_PATH -s SAVE_VIDEO_PATH
(ex) $ python3 mivolo.py --onnx --video input.mp4 --savepath output.mp4
```

<br/>

## Reference

[MiVOLO: Multi-input Transformer for Age and Gender Estimation](https://github.com/WildChlamydia/MiVOLO)

<br/>

## Framework
Pytorch

<br/>

## Model Format
ONNX opset = 18

<br/>

## Netron

[mivolo.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/mivolo/mivolo.onnx.prototxt)  
[yolov8x_person_face.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/mivolo/yolov8x_person_face.onnx.prototxt)
