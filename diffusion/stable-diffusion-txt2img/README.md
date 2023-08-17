# Stable Diffusion - Text-to-Image

## Input

Text to render

- Example
```
a photograph of an astronaut riding a horse
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
$ python3 stable-diffusion-txt2img.py
```

This will save each sample individually as well as a grid of size `--n_iter` x `--n_samples` option values.

If you want to specify the input text, put the text after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 stable-diffusion-txt2img.py --input TEXT --savepath SAVE_IMAGE_PATH
```

Quality, sampling speed and diversity are best controlled via the `--scale`, `--steps` and `--ddim_eta` options.
Higher values of scale produce better samples at the cost of a reduced output diversity.

Furthermore, increasing `--steps` generally also gives higher quality samples, but returns are diminishing for values > 250. Fast sampling (i.e. low values of `--steps`) while retaining good quality can be achieved by using `--ddim_eta` 0.0.

## Change base model

Uses StableDiffusion v1.4 by default. BasilMix and vae-ft-mse can also be used with the command below.

```
python3 stable-diffusion-txt2img.py --sd basil_mix --vae vae-ft-mse --sampler "DPM++ 2M Kerras" -i "masterpiece, best quality, ultra detailed, sketch, oil painting, 1 girl has silver long hair, eyelashes, jewelry eyes, twinklee eyes, glowing eyes, smile, office lady, tight skirt, looking at viewer, outdoor, street, night" --n_prompt "worst quality, low quality, bad anatomy, bad hands, missing arms, text error, missing fingers, jpeg artifacts, long neck, signature, watermark, blurry, fisheye lens, animal, deformed mutated disfigured, mutated hands, missing hands, extra hands, liquid hands, poorly drawn hands, mutated fingers, bad fingers, extra fingers, liquid fingers, poorly drawn fingers, bad legs, missing legs, extra legs, bad arms, extra arms, long torso, thick thighs, partial head, bad face, partial face, bad eyebrows" --steps 20
```

## Reference

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [k-diffusion](https://github.com/crowsonkb/k-diffusion)

## Framework

Pytorch

## Model Format

ONNX opset=12

## Netron

### Legacy version

- [diffusion_emb.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/stable-diffusion-txt2img/diffusion_emb.onnx.prototxt)  
- [diffusion_mid.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/stable-diffusion-txt2img/diffusion_mid.onnx.prototxt)  
- [diffusion_out.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/stable-diffusion-txt2img/diffusion_out.onnx.prototxt)  
- [autoencoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/stable-diffusion-txt2img/autoencoder.onnx.prototxt)

### Re export version

- [diffusion_v1_4.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/stable-diffusion-txt2img/diffusion_v1_4.opt.onnx.prototxt)  
- [autoencoder_v1_4.opt.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/stable-diffusion-txt2img/autoencoder_v1_4.opt.onnx.prototxt)
