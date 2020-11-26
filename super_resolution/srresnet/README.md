# SRResNet

## Input

![Input](lenna.png)

Ailia input shape : (1,3,64,64)  
Range : [0.0, 1.0]

## Output

![Output](output.png)

Ailia output shape : (1,3,256,256)  
Range : [0, 1.0]

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 srresnet.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 srresnet.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 srresnet.py --video VIDEO_PATH
```

The default setting is to use the optimized model and weights, but you can also switch to the normal model by using the `--normal` option.

Image Input only: Instead of resizing input image to (64 * 64) when loading, you can try padding mode using `--padding` option.

## Reference

[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://github.com/twtygqyy/pytorch-SRResNet)

## Framework

Pytorch 1.2.0

## Model Format

ONNX opset = 10

## Netron

[srresnet.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/srresnet/srresnetn.onnx.prototxt)

[srresnet.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/srresnet/srresnet.opt.onnx.prototxt)

