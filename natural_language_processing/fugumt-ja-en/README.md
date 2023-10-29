# FuguMT (ja-en)


## Input

Text (Japanese) to translate


- Example
```
これは猫です
```

## Output

Translated (Japanese) text
```
translation_text: this is a cat
```

## Requirements
This model requires additional module.

```
pip3 install transformers
pip3 install sentencepiece
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample text,
```bash
$ python3 fugumt-ja-en.py
```

If you want to specify the input text, put the text after the `--input` option.
```bash
$ python3 fugumt-ja-en.py --input TEXT
```

## Reference

- [Hugging Face - staka/fugumt-en-ja](https://huggingface.co/staka/fugumt-ja-en)
- [Fugu-Machine Translator](https://github.com/s-taka/fugumt)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[encoder_model.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/fugumt/encoder_model.onnx.prototxt)
[decoder_model.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/fugumt/decoder_model.onnx.prototxt)
