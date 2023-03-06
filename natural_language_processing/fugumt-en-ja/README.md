# FuguMT

## Input

Text (English) to translate

- Example
```
This is a cat.
```

## Output

Translated (Japanese) text
```
translation_text: これは猫です。
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

For the sample image,
```bash
$ python3 fugumt-en-ja.py
```

If you want to specify the input text, put the text after the `--input` option.
```bash
$ python3 fugumt-en-ja.py --input TEXT
```

## Reference

- [Hugging Face - staka/fugumt-en-ja](https://huggingface.co/staka/fugumt-en-ja)
- [Fugu-Machine Translator](https://github.com/s-taka/fugumt)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[seq2seq-lm-with-past.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/fugumt-en-ja/seq2seq-lm-with-past.onnx.prototxt)
