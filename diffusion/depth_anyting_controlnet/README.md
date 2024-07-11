# ControlNet trained on DepthAnything

## Input

* **Image used to control the reverse diffusion process**

![depth image](depth_flower.png)

The script will perform a image generation conditioned on the control image.
Images like this can be generated using the script in `ailia-models/depth_estimation/depth_anything`.

* **Prompt and optional negative prompt**

The default prompt is "Beautiful flower garden full of tulips.", and the negative prompt is set to ""(empty string).

## Output

* **Generated image**

![result](output.png)

## Usage
Internet connection is required when running the script for the first time,
as it will download the necessary model files.

Running this script will generate an image conditioned on the control image.
Use `--help` to see the help message with all the arguments.

#### Example 1: Inference on prepared demo image, with the default prompts and parameters.
```bash
$ python3 depth_anything_controlnet.py
```

#### Example 2: Specify input path and save path.
```bash
$ python3 depth_anything_controlnet.py -i depth_flower.png -s output.png
```
`-i` and `-s` options can be used to specify the
input path and save path separately.

#### Example 3: Specify prompt, negative prompt, and sampler.
```bash
$ python3 depth_anything_controlnet.py --prompt "Flower garden full of yellow tulips" --negative_prompt "Low quality, unrealistic." --n_timesteps 50 --sampler ddim
```
The argument `--n_timesteps` can be used to specify the number of steps the model takes to generate the image.
The sampling method specified in the `--sampler` argument is used in the reverse diffusion process. Currently supports ddpm and ddim only.

#### Example 4: Change the size of the generated image
```bash
$ python3 depth_anything_controlnet.py --width 512 --height 512
```
`--width` and `--height` argument can be used to change the size of the generated image. It may take longer or even be impossible to generate the image depending on the machine being used.

## Reference

- [Depth Anything ](https://github.com/LiheYoung/Depth-Anything)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)

## Framework

Pytorch

## Model Format

ONNX opset=12

## Netron

- [text_encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/depth_anything_controlnet/text_encoder.onnx.prototxt)  
- [unet.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/depth_anything_controlnet/unet.onnx.prototxt)  
- [vae_decoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/depth_anything_controlnet/vae_decoder.onnx.prototxt)