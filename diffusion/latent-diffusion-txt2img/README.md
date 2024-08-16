# Latent Diffusion - Text-to-Image

## Input

Text to render

- Example
```
a virus monster is playing guitar, oil on canvas
```

## Output

![Output](output.png)

## Requirements
This model requires additional module.

```
pip3 install transformers
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 latent-diffusion-txt2img.py
```

This will save each sample individually as well as a grid of size `--n_iter` x `--n_samples` option values.

If you want to specify the input text, put the text after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 latent-diffusion-txt2img.py --input TEXT --savepath SAVE_IMAGE_PATH
```

Quality, sampling speed and diversity are best controlled via the `--scale`, `--ddim_steps` and `--ddim_eta` options.
Higher values of scale produce better samples at the cost of a reduced output diversity.

Furthermore, increasing `--ddim_steps` generally also gives higher quality samples, but returns are diminishing for values > 250. Fast sampling (i.e. low values of `--ddim_steps`) while retaining good quality can be achieved by using `--ddim_eta` 0.0.

## Reference

- [Latent Diffusion Models](https://github.com/CompVis/latent-diffusion)

## Framework

Pytorch

## Model Format

ONNX opset=12

## Netron

[transformer_emb.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/latent-diffusion-txt2img/transformer_emb.onnx.prototxt)  
[transformer_attn.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/latent-diffusion-txt2img/transformer_attn.onnx.prototxt)  
[diffusion_emb.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/latent-diffusion-txt2img/diffusion_emb.onnx.prototxt)  
[diffusion_mid.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/latent-diffusion-txt2img/diffusion_mid.onnx.prototxt)  
[diffusion_out.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/latent-diffusion-txt2img/diffusion_out.onnx.prototxt)  
[autoencoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/latent-diffusion-txt2img/autoencoder.onnx.prototxt)
