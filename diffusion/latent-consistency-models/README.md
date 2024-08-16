# Latent Consistency Models - Text-to-Image

## Input

Text to render

- Example
```
"portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, Tokyo, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography"
```

## Output

![Output](output.png)


## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 latent-consistency-models.py
```

if you are facing error 128, use following command to run your model with cpu:

```bash
$ python3 latent-consistency-models.py -e 1
```



If you want to specify the input text, put the text after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 latent-consistency-models.py --input TEXT --savepath SAVE_IMAGE_PATH
```

Quality, sampling speed and diversity are best controlled via the `--guidance_scale` and `--num_inference_steps`  options.
Higher values of scale produce better samples at the cost of a reduced output diversity.


## Reference

- [Latent Consistency Model](https://github.com/luosiallen/latent-consistency-model)

## Framework

Pytorch

## Model Format

ONNX opset=14

## Netron

[text_encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/latent-consistency-models/text_encoder.onnx.prototxt)  
[unet.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/latent-consistency-models/unet.onnx.prototxt)  
[vae_encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/latent-consistency-models/vae_encoder.onnx.prototxt)  
