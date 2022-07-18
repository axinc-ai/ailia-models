# SwinIR: Image Restoration Using Swin Transformer

## Input
In case of classical model.   
![input](input_classical.png)

(Image from https://github.com/JingyunLiang/SwinIR/tree/main/testsets)

## Output
In case of classical model.    
![Output](output_classical.png)

## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,

``` bash
$ python3 swinir.py --onnx
```

Please be careful that onnxruntime is used bacause ailia model is not implemented.

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.

```bash
$ python3 swinir.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH --onnx
(ex) $ python3 swinir.py --input input_classical.png --savepath example.png --onnx
```

By adding the `--model_name` option, you can choose the model.
```bash
$ python3 swinir.py --model_name MODEL_NAME --onnx
(ex) $ python3 swinir.py --model_name classical --onnx
(ex) $ python3 swinir.py --model_name lightweight --onnx
(ex) $ python3 swinir.py --model_name real --onnx
(ex) $ python3 swinir.py --model_name gray --onnx
(ex) $ python3 swinir.py --model_name color --onnx
(ex) $ python3 swinir.py --model_name jpeg --onnx
```

By adding the --video option, you can input the video.
If you pass 0 as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.

```bash
$ python3 swinir.py --video VIDEO_PATH --onnx
(ex) $ python3 swinir.py --video demo.mp4 --onnx
(ex) $ python3 swinir.py --video demo.mp4 -s output2.mp4 --onnx
(ex) $ python3 swinir.py --video demo.mp4 --model_name classical --onnx
(ex) $ python3 swinir.py --video demo.mp4 -s output.mp4 --model_name lightweight --onnx
```
## Reference

- [(Github) SwinIR: Image Restoration Using Swin Transformer](https://github.com/JingyunLiang/SwinIR)
- [(Paper) SwinIR: Image Restoration Using Swin Transformer](https://arxiv.org/pdf/2108.10257.pdf)

## Framework

Pytorch 1.7.1

## Model Format

ONNX opset=11

## Netron

[001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.onnx.prototxt)
[002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/swinir/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.onnx.prototxt)
[003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.onnx.prototxt)
[004_grayDN_DFWB_s128w8_SwinIR-M_noise25.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/swinir/004_grayDN_DFWB_s128w8_SwinIR-M_noise25.onnx.prototxt)
[005_colorDN_DFWB_s128w8_SwinIR-M_noise25.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.onnx.prototxt)
[006_CAR_DFWB_s126w7_SwinIR-M_jpeg10.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/swinir/006_CAR_DFWB_s126w7_SwinIR-M_jpeg10.onnx.prototxt)
