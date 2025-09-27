# ruri-v3-310m 

## Input

TEXT file.

## Output

The sentence closest to the input prompt.

## Requirements

This model requires additional modules.

```
pip3 install ailia_tokenizer
```

## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample text,
```bash
$ python3 ruri-v3.py
```

If you want to specify the text or pdf file, put the file path after the `-i` option.  
```bash
$ python3 ruri-v3.py -i FILE_PATH
```

## Example

```
User (press q to exit): nnapiの速度
Text: NNAPIを使用することで、Google PixelのEdgeTPUや、QualcommやMediatekのNPUを 使用した高速推論が可能になります。 (Similarity:0.863)
```

## Reference

- [Hugging Face - ruri-v3-310m ](https://huggingface.co/cl-nagoya/ruri-v3-310m)

## Framework

Pytorch

## Model Format

ONNX opset=16

## Netron

[ruri-v3-310m.onnx.prototxt](https://storage.googleapis.com/ailia-models/ruri-v3/ruri-v3-310m.onnx.prototxt)  
