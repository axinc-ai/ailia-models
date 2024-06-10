# LLaVA

## Input

- image

  <img src="view.jpg" height="256px"/>

  (Image from https://llava-vl.github.io/static/images/view.jpg)

- prompt

  What are the things I should be cautious about when I visit here?

## Output

When visiting this location, which features a pier extending over a large body of water, you should be cautious about several things. First, be mindful of the weather conditions, as the pier may be affected by strong winds or storms, which could make it unsafe to walk on. Second, be aware of the water depth and currents, as they can change rapidly and pose a risk to swimmers or those who venture too close to the edge. Additionally, be cautious of the presence of any wildlife in the area, as they may pose a potential danger or distraction. Finally, be mindful of the pier's structural integrity, as it may be subject to wear and tear over time, and it is essential to ensure that it is safe for use.

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
$ python3 llava.py
```

If you want to specify the input image, put the image path after the `--input` option.  
You can use `--savepath` option to change the name of the output file to save.
```bash
$ python3 llava.py --input IMAGE_PATH --savepath SAVE_IMAGE_PATH
```

If you want to specify the input prompt, specify it after the `--prompt` option.
```bash
$ python3 llava.py --prompt PROMPT
```

## Reference

- [LLaVA](https://github.com/haotian-liu/LLaVA)

## Framework

Pytorch

## Model Format

ONNX opset=14

## Netron

[llava-v1.5-7b.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llava/llava-v1.5-7b.onnx.prototxt)  
[encode_images.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llava/encode_images.onnx.prototxt)  
[embed_tokens.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llava/embed_tokens.onnx.prototxt)
