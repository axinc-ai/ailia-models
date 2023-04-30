# Sentence Transformers Japanese

## Input

PDF file.

## Output

The sentence closest to the input prompt.

## Requirements
This model requires additional module.

```
pip3 install pdfminer.six
```

## Usage

Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample pdf,
```bash
$ python3 sentence_transformer_japanese.py
```

If you want to specify the pdf file, put the file path after the `--file` option.  
```bash
$ python3 sentence_transformer_japanese.py --file PDF_FILE_PATH
```

## Reference

- [sentence-transformers](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)

## Framework

Pytorch

## Model Format

ONNX opset=11

## Netron

[paraphrase-multilingual-mpnet-base-v2.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/sentence-transformers-japanese/paraphrase-multilingual-mpnet-base-v2.onnx.prototxt)  
