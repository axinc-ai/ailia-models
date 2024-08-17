# bert-ner-japanese

## Input

A `SENTENCE`.

- Example
```
株式会社Jurabiは、東京都台東区に本社を置くIT企業である。
```

## Output

NER(named entity recognition)
```
[
  {
    "entity_group": "法人名",
    "score": 0.9949989716211954,
    "word": "株式 会社 Jurabi",
    "start": null,
    "end": null
  },
  {
    "entity_group": "地名",
    "score": 0.996607705950737,
    "word": "東京 都 台東 区",
    "start": null,
    "end": null
  }
]
```


## Requirements

This model requires additional module.
```
pip3 install transformers
```

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample text,
```bash
$ python3 bert_ner_japanese.py
```

If you want to specify the input text, put the text after the `--input` option.
```bash
$ python3 bert_ner_japanese.py --input TEXT
```


## Reference

- [Hugging Face - jurabi/bert-ner-japanese](https://huggingface.co/jurabi/bert-ner-japanese)

## Framework

Pytorch

## Model Format

ONNX opset=17

## Netron

[model.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/bert_ner_japanese/model.onnx.prototxt)  
