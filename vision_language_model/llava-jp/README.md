# LLaVA-JP

## Input

- Image

  ![Input](sample.jpg)

  (Image from https://huggingface.co/rinna/bilingual-gpt-neox-4b-minigpt4/blob/main/sample.jpg)

- Prompt

  猫の隣には何がありますか？

## Output

猫はノートパソコンの横に座っています。

## Requirements

This model requires additional module.

```
pip install transformers
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample image,
```bash
$ python3 llava-jp.py
```

If you want to specify the input image, put the image path after the `--input` option.  
```bash
$ python3 llava-jp.py --input IMAGE_PATH --prompt "猫の隣には何がありますか？"
```

If you want to specify the prompt, put the prompt after the `--prompt` option.  
```bash
$ python3 llava-jp.py --prompt PROMPT
```

## Reference

- [LLaVA-JP](https://github.com/tosiyuki/LLaVA-JP/tree/main)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

[llava-jp-1.3b-v1.1.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llava-jp/llava-jp-1.3b-v1.1.onnx.prototxt)  
[encode_images.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/llava-jp/encode_images.onnx.prototxt)  
