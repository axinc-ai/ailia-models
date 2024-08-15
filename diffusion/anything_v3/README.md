# Anything V3

## Input

Text to render

Example
```
witch
```

## Output

![Output](output.png)

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

Since ailia doesn't support anything v3.0 due to file size of weights, you have to use `--onnx` option.

For the sample image,
```bash
$ python3 anything_v3.py --onnx
```

If you want to specify the input text, put the text after the `--input` option.  
```bash
$ python3 anything_v3 --onnx --input TEXT
```

## Reference
[Linaqruf/anything-v3.0](https://huggingface.co/Linaqruf/anything-v3.0)  

[creativeml-openrail-m](https://huggingface.co/spaces/CompVis/stable-diffusion-license)

## Framework

Python 3.9.12    
Pytorch 1.12.0    
transformers 4.25.1   
diffusers 0.11.0 

## Model Format

ONNX opset=14

## Netron

[safety_checker.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/anything_v3/safety_checker.onnx.prototxt)  

[text_encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/anything_v3/text_encoder) 

[unet.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/anything_v3/unet.onnx.prototxt)  

[vae_decoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/anything_v3/vae_decoder.onnx.prototxt)  

[vae_encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/anything_v3/vae_encoder.onnx.prototxt)  
