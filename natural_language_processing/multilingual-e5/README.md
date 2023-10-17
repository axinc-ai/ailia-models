# Multilingual-E5

## Input

TEXT or PDF file.

## Output

The sentence closest to the input prompt.

## Requirements

This model requires additional module if you want to load pdf file.

```
pip3 install transformers
pip3 install pdfminer.six
```

## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample text,
```bash
$ python3 multilingual-e5.py
```

If you want to specify the text or pdf file, put the file path after the `-i` option.  
```bash
$ python3 multilingual-e5.py -i FILE_PATH
```

## Example

```
User (press q to exit): nnapiの速度
Text: NNAPIを使用することで、Google PixelのEdgeTPUや、QualcommやMediatekのNPUを 使用した高速推論が可能になります。 (Similarity:0.863)
```

## Reference

- [Hugging Face - Multilingual-E5-base](https://huggingface.co/intfloat/multilingual-e5-base)
- [Hugging Face - Multilingual-E5-large](https://huggingface.co/intfloat/multilingual-e5-large)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[multilingual-e5-base.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/multilingual-e5/multilingual-e5-base.onnx.prototxt)  
[multilingual-e5-large.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/multilingual-e5/multilingual-e5-large.onnx.prototxt)  
