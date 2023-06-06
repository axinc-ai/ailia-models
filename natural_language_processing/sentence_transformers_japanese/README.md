# Sentence Transformers Japanese

## Input

TEXT or PDF file.

## Output

The sentence closest to the input prompt.

## Requirements
This model requires additional module if you want to load pdf file.

```
pip3 install pdfminer.six
```

## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample text,
```bash
$ python3 sentence_transformer_japanese.py
```

If you want to specify the text or pdf file, put the file path after the `-i` option.  
```bash
$ python3 sentence_transformer_japanese.py -i FILE_PATH
```

## Example

```
User (press q to exit): nnapiの速度
Text: 実際、弊社でもSnapdragon 8+ Gen1とyolox_tinyにおいて、CPU（float）に比べてNNAPI NPU（int8）で15倍高速に動作することを確認しています。 (Similarity:0.592)
```

## Reference

- [sentence-transformers](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[paraphrase-multilingual-mpnet-base-v2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/sentence-transformers-japanese/paraphrase-multilingual-mpnet-base-v2.onnx.prototxt)  
