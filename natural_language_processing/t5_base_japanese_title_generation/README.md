# T5 Base Japanese Title Generation

## Input

TEXT file.

## Output

The title appropriate to the input prompt.


## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample text,
```bash
$ python3 t5_base_japanese_title_generation.py
```

If you want to specify the text or pdf file, put the file path after the `-i` option.  
```bash
$ python3 t5_base_japanese_title_generation.py -i FILE_PATH
```

## Reference

- [t5-japanese](https://github.com/sonoisa/t5-japanese)

### Framework
PyTorch

## Model Format

ONNX opset=12

## Netron

### encoder
[t5-base-japanese-title-generation-encoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/t5_base_japanese_title_generation/t5-base-japanese-title-generation-encoder.onnx.prototxt)

### decoder
[t5-base-japanese-title-generation-decoder.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/t5_base_japanese_title_generation/t5-base-japanese-title-generation-decoder-with-lm-head.onnx.prototxt)
