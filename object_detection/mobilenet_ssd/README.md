# Mobilenet_SSD

### input

![Input](input.jpg)

(Image from https://pixabay.com/ja/photos/%E3%83%AD%E3%83%B3%E3%83%89%E3%83%B3%E5%B8%82-%E9%8A%80%E8%A1%8C-%E3%83%AD%E3%83%B3%E3%83%89%E3%83%B3-4481399/)

Ailia input shape(1, 3, 300, 300)  
Range:[0, 1]

### output

![output_image](annotated.png)


### usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 mobilenet_ssd.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 mobilenet_ssd.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 mobilenet_ssd.py --video VIDEO_PATH
```

You can select a pretrained model by specifying `-a mb1-ssd` or `-a mb2-ssd-lite`(default).


### Reference

[MobileNetV1, MobileNetV2, VGG based SSD/SSD-lite implementation in Pytorch](https://github.com/qfgaohao/pytorch-ssd)


### Framework
PyTorch 1.0 / 0.4


### Model Format
ONNX opset = 10


### Netron

[mb2-ssd-lite.onnx.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/mobilenet_ssd/mb2-ssd-lite.onnx.prototxt)

