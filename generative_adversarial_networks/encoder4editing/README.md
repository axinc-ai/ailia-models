# encoder4editing

## Input

<img src="demo.jpg" width="256">  

(Image from https://github.com/omertov/encoder4editing/tree/main/notebooks/images)

## Output

- inversion

  <img src="output.png" width="256">  

- InterFaceGAN

  age, smile and pose directions for the FFHQ StyleGAN Generator.

  <img src="example/age_edit.png">  

- GANSpace

  editings for the cars domain, as well as several examples for the facial domain taken from the official GANSpace repository.
  
  - ffhq
  <img src="example/ffhq_edit.png">
  
  - cars
  <img src="example/car_edit.png">  

- SeFa

  apply to the selected editing parameters.
  <img src="example/ffhq_sefa.png">

## Requirements
This model requires additional module.

```
pip3 install dlib     # for align face
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 encoder4editing.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 encoder4editing.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

By adding the `--model_type` option, you can specify model type which is selected from "ffhq", "car", "horse", "church". (default is ffhq)
```bash
$ python3 encoder4editing.py --model_type ffhq
```

By adding the `--model_type` option, you can specify model type which is selected from "ffhq", "car", "horse", "church". (default is ffhq)
```bash
$ python3 encoder4editing.py --model_type ffhq
```

## Reference

- [Designing an Encoder for StyleGAN Image Manipulation](https://github.com/omertov/encoder4editing)

## Framework

Pytorch

## Model Format

ONNX opset=12

## Netron

[ffhq_encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/encoder4editing/ffhq_encoder.onnx.prototxt)  
[cars_encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/encoder4editing/carsencoder.onnx.prototxt)  
[horse_encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/encoder4editing/horse_encoder.onnx.prototxt)  
[church_encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/encoder4editing/church_encoder.onnx.prototxt)

[ffhq_decoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/encoder4editing/ffhq_encoder.onnx.prototxt)  
[cars_decoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/encoder4editing/cars_decoder.onnx.prototxt)  
[horse_decoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/encoder4editing/horse_decoder.onnx.prototxt)  
[church_decoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/encoder4editing/church_decoder.onnx.prototxt)
