# Invertible Denoising Network

## Input

![Input](./sample/input_1_09.PNG)

input image/video size 256×256

## Output

![Output](./sample/output_1_09.PNG)

output image/video size 256×256

### usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

Because ailia does not support InvDN model, you must add `--onnx` option when you run this sample.

In order to specify predict mode, you must add `--ftype` option.
1. image    
If you predict denoised image, you specify `--ftype` option and `--input` as below.  
``` bash
python3 invertible_denoising_network.py --onnx --ftype=image --input="input_images/1_00.PNG"
```
1. video    
If you predict denoised video, you specify `--ftype` option and `--input` as below. Please be careful that an input video is noised at first and the model denoised this.    
``` bash
python3 invertible_denoising_network.py --onnx --ftype=video --input="input_videos/video.mp4"
```
1. webcamera video      
By adding the `--video` option, you can use the webcam input instead of the video file.  
``` bash
python3 invertible_denoising_network.py --onnx --ftype=video --video=0 --outname="test"
```

## Reference
- Repository    
[Invertible Image Denoising](https://github.com/Yang-Liu1082/InvDN)

- Input images    
SIDD Benchmark > Download > SIDD Benchmark Data > Noisy sRGB data    
[Smartphone Image Denoising Dataset](https://www.eecs.yorku.ca/~kamel/sidd/benchmark.php)

## Framework

Pytorch 1.5.0

## Model Format

ONNX opset = 11

## Netron

[InvDN.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/invertible_denoising_network/InvDN.onnx.prototxt)
