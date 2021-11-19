# Anime GAN v2

## Input

<img src="sample.jpg" width="512px">

(Image from https://github.com/bryandlee/animegan2-pytorch/blob/main/samples/inputs/3.jpg)

## Output

<img src="output.png" width="512px">

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
``` bash
$ python3 animeganv2.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 animeganv2.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--video` option, you can input the video.   
If you pass `0` as an argument to VIDEO_PATH, you can use the webcam input instead of the video file.
```bash
$ python3 animeganv2.py --video VIDEO_PATH
```

By adding the `--model_name` option, you can specify model type which is selected from "paprika", "hayao", "shinkai", "celeba", "face_paint". (default is paprika)
```bash
$ python3 animeganv2.py --model_name paprika
```

## Reference

- [PyTorch Implementation of AnimeGANv2](https://github.com/bryandlee/animegan2-pytorch)
- [AnimeGANv2](https://github.com/TachibanaYoshino/AnimeGANv2)

## Framework

Tensorflow

## Model Format

ONNX opset=11

## Netron

[generator_Paprika.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/animeganv2/generator_Paprika.onnx.prototxt)  
[generator_Hayao.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/animeganv2/generator_Hayao.onnx.prototxt)  
[generator_Shinkai.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/animeganv2/generator_Shinkai.onnx.prototxt)  
[celeba_distill.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/animeganv2/celeba_distill.onnx.prototxt)  
[face_paint_512_v2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/animeganv2/face_paint_512_v2.onnx.prototxt)
