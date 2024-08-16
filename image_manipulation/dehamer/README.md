# Dehamer

## Input

![Input](canyon.png)

(Image from https://github.com/Li-Chongyi/Dehamer/blob/main/data/classic_test_image/input/canyon.png)

Shape : (1, 3, 960, 960)  

## Output

![Output](output.png)

Shape : (1, 3, 960, 960)  

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 dehamer.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 dehamer.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 dehamer.py --video VIDEO_PATH
```

By adding the `--model_type` option, you can specify model type which is selected from "NH", "dense", "indoor", "outdoor". (default is NH)
```bash
$ python3 dehamer.py --model_type outdoor
```

## Reference

- [Dehamer](https://github.com/Li-Chongyi/Dehamer)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[PSNR2066_SSIM06844_NH.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/dehamer/PSNR2066_SSIM06844_NH.onnx.prototxt)  
[PSNR3518_SSIM09860_outdoor.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/dehamer/PSNR3518_SSIM09860_outdoor.onnx.prototxt)  
[PSNR1662_SSIM05602_dense.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/dehamer/PSNR1662_SSIM05602_dense.onnx.prototxt)  
[PSNR3663_ssim09881_indoor.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/dehamer/PSNR3663_ssim09881_indoor.onnx.prototxt)
