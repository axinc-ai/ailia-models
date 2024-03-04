# Named Entity Recognition Using Bert

### input
A `SENTENCE`.

### output
Whether the word is a named entity.

|Label|Detail|
|---|---|
|O|Outside of a named entity B-MIS|
|B-MIS|Beginning of a miscellaneous entity right after another miscellaneous entity|
|I-MIS|Miscellaneous entity|
|B-PER|Beginning of a person’s name right after another person’s name|
|I-PER|Person’s name|
|B-ORG|Beginning of an organisation right after another organisation|
|I-ORG|Organisation|
|B-LOC|eginning of a location right after another location|
|I-LOC|Location|

### Usage
Set the `SENTENCE` as an argument.

```bash
$ python3 bert_ner.py -i "My name is bert"
...
Input :  My name is bert
Output :  [{'word': 'be', 'score': 0.9467903971672058, 'entity': 'I-PER', 'index': 4}, {'word': '##rt', 'score': 0.8386904001235962, 'entity': 'I-PER', 'index': 5}]
```

### Reference
[transformers](https://github.com/huggingface/transformers)  

[onnx-transformers](https://github.com/patil-suraj/onnx_transformers)  

### Framework
PyTorch 1.6.0

### Model Format
ONNX opset = 11

### Netron

- [bert-large-cased-finetuned-conll03-english.onnx.prototxt](https://netron.app/?url=https://storage.googleapis.com/ailia-models/bert_ner/bert-large-cased-finetuned-conll03-english.onnx.prototxt)
